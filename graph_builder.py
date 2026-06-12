"""
graph_builder.py
----------------
LLM ile metinden varlık/ilişki çıkarır ve NetworkX DiGraph oluşturur.
Kurumsal düzeyde hata yönetimi ve loglama içerir.
"""

import json
import logging
import re
from typing import Any, Dict, List, Tuple

import networkx as nx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

# ─── Sabitler ────────────────────────────────────────────────────────────────

ENTITY_RELATION_SYSTEM_PROMPT = """Sen, verilen metin parçasından yapılandırılmış bilgi çıkaran bir uzman sistemsin.
Görevin: Metindeki önemli varlıkları (kişiler, kuruluşlar, ürünler, kavramlar, tarihler, yerler)
ve bu varlıklar arasındaki ilişkileri tespit ederek YALNIZCA aşağıdaki JSON formatında döndürmektir.

ÇIKTI FORMATI (başka hiçbir şey ekleme):
{
  "entities": [
    {"id": "BenzersizID", "label": "Varlık Adı", "type": "PERSON|ORG|PRODUCT|CONCEPT|DATE|PLACE"}
  ],
  "relations": [
    {"source": "KaynakID", "target": "HedefID", "relation": "ilişki_türü"}
  ]
}

KURALLAR:
- ID'ler boşluksuz, küçük harfli ve Türkçe karakter içermez (örn: "alan_turing", "ibm", "deep_blue").
- Varlık adları orijinal metindeki gibi yazılır.
- İlişki türleri kısa ve açıklayıcı olur (örn: "geliştirdi", "kurdu", "yendi", "çalışıyor").
- Sadece JSON döndür, Markdown bloğu veya açıklama ekleme.
"""


# ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────

def _clean_json_response(raw: str) -> str:
    """
    LLM yanıtından Markdown kod bloklarını ve fazla boşlukları temizler.

    Args:
        raw: Ham LLM yanıt metni.

    Returns:
        Temizlenmiş JSON string.
    """
    # Markdown ```json ... ``` bloklarını temizle
    cleaned = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()
    return cleaned


def _safe_parse_json(raw: str, chunk_index: int) -> Dict[str, Any]:
    """
    JSON parsing işlemini güvenli şekilde yapar.

    Args:
        raw: Parse edilecek ham string.
        chunk_index: Hata mesajı için chunk sırası.

    Returns:
        Parse edilmiş dict ya da boş yapı.
    """
    cleaned = _clean_json_response(raw)
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Yanıt bir dict değil.")
        data.setdefault("entities", [])
        data.setdefault("relations", [])
        return data
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Chunk %d için JSON parse hatası: %s — Ham yanıt: %.200s",
            chunk_index,
            exc,
            cleaned,
        )
        return {"entities": [], "relations": []}


# ─── Ana Sınıf ────────────────────────────────────────────────────────────────

class GraphBuilder:
    """
    LLM kullanarak metinden bilgi grafiği oluşturur.

    Attributes:
        llm: LangChain uyumlu LLM nesnesi.
        graph: NetworkX yönlü grafiği (DiGraph).
    """

    def __init__(self, llm: ChatGroq):
        """
        Args:
            llm: Başlatılmış ChatGroq (veya uyumlu) LLM nesnesi.
        """
        self.llm = llm
        self.graph: nx.DiGraph = nx.DiGraph()
        logger.debug("GraphBuilder başlatıldı.")

    # ── LLM ile çıkarım ──────────────────────────────────────────────────────

    def extract_entities_and_relations(
        self, chunk: str, chunk_index: int = 0
    ) -> Dict[str, Any]:
        """
        Tek bir metin parçasından varlık ve ilişkileri LLM ile çıkarır.

        Args:
            chunk: İşlenecek metin parçası.
            chunk_index: Loglama amacıyla chunk sırası.

        Returns:
            'entities' ve 'relations' listelerini içeren dict.
        """
        logger.debug("Chunk %d için LLM çıkarımı başlıyor.", chunk_index)
        messages = [
            SystemMessage(content=ENTITY_RELATION_SYSTEM_PROMPT),
            HumanMessage(content=f"Metin:\n{chunk}"),
        ]
        try:
            response = self.llm.invoke(messages)
            raw_content = response.content
            logger.debug(
                "Chunk %d LLM yanıtı alındı (%d karakter).",
                chunk_index,
                len(raw_content),
            )
            return _safe_parse_json(raw_content, chunk_index)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Chunk %d için LLM çağrısı başarısız: %s", chunk_index, exc
            )
            return {"entities": [], "relations": []}

    # ── Graf oluşturma ────────────────────────────────────────────────────────

    def _add_entities_to_graph(self, entities: List[Dict[str, Any]]) -> int:
        """
        Varlık listesini NetworkX graf düğümlerine ekler.

        Args:
            entities: Her biri 'id', 'label', 'type' içeren dict listesi.

        Returns:
            Eklenen yeni düğüm sayısı.
        """
        added = 0
        for entity in entities:
            node_id = entity.get("id", "").strip()
            label = entity.get("label", node_id).strip()
            etype = entity.get("type", "UNKNOWN").strip()

            if not node_id:
                logger.warning("Boş ID'li varlık atlandı: %s", entity)
                continue

            if not self.graph.has_node(node_id):
                self.graph.add_node(node_id, label=label, type=etype)
                added += 1
                logger.debug("Düğüm eklendi: '%s' (%s)", label, etype)
            else:
                # Mevcut düğümün etiketini koruyarak sadece type'ı güncelle
                if etype != "UNKNOWN":
                    self.graph.nodes[node_id]["type"] = etype
        return added

    def _add_relations_to_graph(self, relations: List[Dict[str, Any]]) -> int:
        """
        İlişki listesini NetworkX graf kenarlarına ekler.

        Args:
            relations: Her biri 'source', 'target', 'relation' içeren dict listesi.

        Returns:
            Eklenen yeni kenar sayısı.
        """
        added = 0
        for rel in relations:
            source = rel.get("source", "").strip()
            target = rel.get("target", "").strip()
            relation = rel.get("relation", "ilişkili").strip()

            if not source or not target:
                logger.warning("Eksik kaynak/hedef ile ilişki atlandı: %s", rel)
                continue

            # Düğümler yoksa otomatik ekle (fallback)
            for node in (source, target):
                if not self.graph.has_node(node):
                    self.graph.add_node(node, label=node, type="UNKNOWN")
                    logger.debug("Eksik düğüm eklendi (fallback): '%s'", node)

            if not self.graph.has_edge(source, target):
                self.graph.add_edge(source, target, relation=relation)
                added += 1
                logger.debug(
                    "Kenar eklendi: '%s' --[%s]--> '%s'",
                    source,
                    relation,
                    target,
                )
            else:
                # Paralel ilişkiyi güncelleme yerine birleştir
                existing = self.graph[source][target].get("relation", "")
                if relation not in existing:
                    self.graph[source][target]["relation"] = (
                        f"{existing}, {relation}".strip(", ")
                    )
        return added

    def build_from_chunks(self, chunks: List[str]) -> nx.DiGraph:
        """
        Tüm chunk listesinden bilgi grafiğini oluşturur.

        Args:
            chunks: DocumentProcessor'dan gelen metin parçaları.

        Returns:
            Doldurulan NetworkX DiGraph nesnesi.
        """
        total_nodes = 0
        total_edges = 0

        for idx, chunk in enumerate(chunks, start=1):
            logger.info(
                "Graf oluşturma: chunk %d/%d işleniyor...", idx, len(chunks)
            )
            extracted = self.extract_entities_and_relations(chunk, idx)

            new_nodes = self._add_entities_to_graph(extracted.get("entities", []))
            new_edges = self._add_relations_to_graph(extracted.get("relations", []))

            total_nodes += new_nodes
            total_edges += new_edges
            logger.info(
                "Chunk %d: +%d düğüm, +%d kenar eklendi.", idx, new_nodes, new_edges
            )

        logger.info(
            "Graf oluşturma tamamlandı. Toplam: %d düğüm, %d kenar.",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph

    # ── İstatistik ───────────────────────────────────────────────────────────

    def get_graph_summary(self) -> Dict[str, Any]:
        """Graf hakkında özet istatistikler döndürür."""
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "is_directed": self.graph.is_directed(),
            "nodes": [
                {
                    "id": n,
                    "label": self.graph.nodes[n].get("label", n),
                    "type": self.graph.nodes[n].get("type", "UNKNOWN"),
                }
                for n in self.graph.nodes
            ],
        }

    def get_graph(self) -> nx.DiGraph:
        """Mevcut NetworkX DiGraph nesnesini döndürür."""
        return self.graph

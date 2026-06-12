"""
query_engine.py
---------------
Kullanıcı sorgularını bilgi grafiği üzerinden işleyip yanıt üretir.
Kurumsal düzeyde hata yönetimi ve loglama içerir.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

# ─── Sabitler ────────────────────────────────────────────────────────────────

QUERY_SYSTEM_PROMPT = """Sen, verilen Bilgi Grafiği bağlamını kullanarak soruları yanıtlayan bir uzman yapay zeka asistanısın.

KURALLARIN:
1. Yalnızca verilen Bilgi Grafiği bağlamındaki bilgileri kullan.
2. Eğer bağlamda bilgi yoksa, bunu açıkça belirt.
3. Yanıtları Türkçe ver, açık ve sade bir dil kullan.
4. Önemli varlıkların ve ilişkilerin adlarını yanıtta vurgula.
5. Yanıtın sonunda kısa bir özet ver.
"""

QUERY_HUMAN_TEMPLATE = """Bilgi Grafiği Bağlamı:
{context}

Kullanıcı Sorusu: {question}

Lütfen yalnızca bu bağlamı kullanarak soruyu kapsamlı biçimde yanıtla."""


# ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────

def _normalize_query(query: str) -> str:
    """Sorguyu küçük harfe çevirir ve fazla boşlukları temizler."""
    return " ".join(query.lower().split())


def _find_matching_nodes(
    graph: nx.DiGraph, query_tokens: List[str]
) -> List[str]:
    """
    Sorgu tokenlarıyla eşleşen graf düğümlerini bulur (kısmi eşleşme).

    Args:
        graph: Aranacak NetworkX DiGraph.
        query_tokens: Sorgudan türetilmiş kelime listesi.

    Returns:
        Eşleşen düğüm ID'lerinin listesi.
    """
    matched: List[str] = []
    for node_id in graph.nodes:
        node_data = graph.nodes[node_id]
        node_label = node_data.get("label", node_id).lower()
        node_id_lower = node_id.lower()

        for token in query_tokens:
            if len(token) < 3:
                continue  # Çok kısa tokenleri atla
            if token in node_label or token in node_id_lower:
                matched.append(node_id)
                break

    logger.debug("Sorgu ile eşleşen düğümler: %s", matched)
    return matched


def _build_subgraph_context(
    graph: nx.DiGraph,
    seed_nodes: List[str],
    hop_depth: int = 2,
) -> Tuple[str, Set[str]]:
    """
    Başlangıç düğümlerinden itibaren verilen derinlikte komşuları bulur
    ve bağlamı metin olarak döndürür.

    Args:
        graph: Ana NetworkX DiGraph.
        seed_nodes: Başlangıç düğüm ID'leri.
        hop_depth: Kaç hop uzağa gidileceği.

    Returns:
        Tuple[str, Set[str]]: Bağlam metni ve ziyaret edilen düğüm kümesi.
    """
    visited: Set[str] = set()
    context_lines: List[str] = []

    def _explore(node: str, depth: int) -> None:
        if node in visited or depth == 0:
            return
        visited.add(node)
        label = graph.nodes[node].get("label", node)
        ntype = graph.nodes[node].get("type", "UNKNOWN")
        context_lines.append(f"• Varlık: {label} (Tür: {ntype})")

        # Giden kenarlar
        for _, neighbor, edge_data in graph.out_edges(node, data=True):
            relation = edge_data.get("relation", "ilişkili")
            neighbor_label = graph.nodes[neighbor].get("label", neighbor)
            context_lines.append(
                f"  → '{label}' --[{relation}]--> '{neighbor_label}'"
            )
            _explore(neighbor, depth - 1)

        # Gelen kenarlar
        for predecessor, _, edge_data in graph.in_edges(node, data=True):
            if predecessor in visited:
                continue
            relation = edge_data.get("relation", "ilişkili")
            pred_label = graph.nodes[predecessor].get("label", predecessor)
            context_lines.append(
                f"  ← '{pred_label}' --[{relation}]--> '{label}'"
            )

    for seed in seed_nodes:
        if seed in graph.nodes:
            _explore(seed, hop_depth)

    context = "\n".join(context_lines) if context_lines else "Bağlam bulunamadı."
    logger.debug(
        "%d seed düğümden %d satır bağlam oluşturuldu.",
        len(seed_nodes),
        len(context_lines),
    )
    return context, visited


# ─── Ana Sınıf ────────────────────────────────────────────────────────────────

class QueryEngine:
    """
    Kullanıcı sorgularını Bilgi Grafiği üzerinden işler ve yanıt üretir.

    Attributes:
        llm: LangChain uyumlu LLM nesnesi.
        graph: Sorgulama yapılacak NetworkX DiGraph.
        hop_depth: Subgraf genişletme derinliği.
    """

    def __init__(
        self,
        llm: ChatGroq,
        graph: nx.DiGraph,
        hop_depth: int = 2,
    ):
        """
        Args:
            llm: Başlatılmış ChatGroq nesnesi.
            graph: GraphBuilder tarafından oluşturulmuş DiGraph.
            hop_depth: Subgraf keşif derinliği (varsayılan: 2).
        """
        if graph is None or not isinstance(graph, nx.DiGraph):
            raise TypeError("Geçerli bir NetworkX DiGraph sağlanmalıdır.")
        self.llm = llm
        self.graph = graph
        self.hop_depth = hop_depth
        logger.debug(
            "QueryEngine başlatıldı: %d düğüm, %d kenar, hop_depth=%d",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            hop_depth,
        )

    def _identify_relevant_nodes(self, query: str) -> List[str]:
        """
        Sorgu metnindeki anahtar kelimeleri graf düğümleriyle eşleştirir.

        Args:
            query: Kullanıcı sorgu metni.

        Returns:
            İlgili düğüm ID listesi.
        """
        normalized = _normalize_query(query)
        tokens = normalized.split()
        matched = _find_matching_nodes(self.graph, tokens)

        if not matched:
            logger.info(
                "Sorgu için doğrudan eşleşme bulunamadı, tüm düğümler denenecek."
            )
            # Fallback: tüm düğümleri döndür (bağlam kısaltılacak)
            matched = list(self.graph.nodes)[:10]

        logger.info("Sorgu için %d ilgili düğüm tespit edildi.", len(matched))
        return matched

    def _build_context(self, query: str) -> Tuple[str, int]:
        """
        Sorgu için Bilgi Grafiği bağlamını oluşturur.

        Args:
            query: Kullanıcı sorgu metni.

        Returns:
            Tuple[str, int]: Bağlam metni ve kullanılan düğüm sayısı.
        """
        relevant_nodes = self._identify_relevant_nodes(query)
        context, visited = _build_subgraph_context(
            self.graph, relevant_nodes, self.hop_depth
        )
        return context, len(visited)

    def answer(self, query: str) -> Dict[str, Any]:
        """
        Kullanıcı sorgusunu yanıtlar.

        Args:
            query: Kullanıcı sorusu.

        Returns:
            'answer', 'context_node_count', 'matched_nodes' anahtarlarını
            içeren dict.
        """
        if not query or not query.strip():
            logger.warning("Boş sorgu alındı.")
            return {
                "answer": "Lütfen geçerli bir soru girin.",
                "context_node_count": 0,
                "matched_nodes": [],
            }

        logger.info("Sorgu işleniyor: '%s'", query)

        # Bağlam oluştur
        context, node_count = self._build_context(query)
        matched_nodes = self._identify_relevant_nodes(query)

        logger.debug("Oluşturulan bağlam:\n%s", context)

        # LLM'e sor
        human_content = QUERY_HUMAN_TEMPLATE.format(
            context=context, question=query
        )
        messages = [
            SystemMessage(content=QUERY_SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        try:
            response = self.llm.invoke(messages)
            answer_text = response.content.strip()
            logger.info(
                "Sorgu yanıtlandı. Bağlam düğüm sayısı: %d", node_count
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM sorgu çağrısı başarısız: %s", exc)
            answer_text = (
                f"Yanıt üretilirken bir hata oluştu: {exc}\n"
                "Lütfen API anahtarınızı ve bağlantınızı kontrol edin."
            )

        return {
            "answer": answer_text,
            "context_node_count": node_count,
            "matched_nodes": matched_nodes,
        }

    def get_graph_stats(self) -> Dict[str, Any]:
        """Graf istatistiklerini döndürür."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "is_directed": self.graph.is_directed(),
        }

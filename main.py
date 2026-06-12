"""
main.py
-------
GraphRAG CLI uygulamasi - ana orkestrasyon ve arayuz modulu.
Kullanim: python main.py [--file PATH] [--interactive] [--query "SORU"]
"""

import argparse
import io
import logging
import os
import sys

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from document_processor import DocumentProcessor
from graph_builder import GraphBuilder
from query_engine import QueryEngine

# Windows konsolunda UTF-8 cikti icin stdout'u yeniden yapilandir
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
    except AttributeError:
        pass  # Interactive modlarda buffer olmayabilir


# --------------------------------------------------------------------------- #
#  Loglama Konfigurasyonu
# --------------------------------------------------------------------------- #

def setup_logging(level: str = "INFO") -> None:
    """
    Uygulama genelinde loglama sistemini yapilandirir.

    Args:
        level: Log seviyesi ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("graphrag.log", encoding="utf-8"),
        ],
    )


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Banner (ASCII-only, Windows uyumlu)
# --------------------------------------------------------------------------- #

BANNER = """
+==================================================================+
|                                                                  |
|   ____  ____   __    ____  _  _  ____   __    ____              |
|  / ___)(  _ \\ / _\\  (  _ \\( \\/ )(  _ \\ / _\\  / ___)             |
|  \\___ \\ ) __//    \\  ) __/ )  /  )   //    \\ \\___ \\             |
|  (____/(__)  \\_/\\_/(__)  (__/  (__\\_)\\_/\\_/ (____/             |
|                                  R  A  G                         |
|                                                                  |
|     Bilgi Grafigi Tabanli Otonom RAG Sistemi v1.0                |
|     LangChain + Groq (Llama-3.3-70B-Versatile) + NetworkX       |
+==================================================================+
"""


# --------------------------------------------------------------------------- #
#  Ortam Degiskeni Yukleme
# --------------------------------------------------------------------------- #

def load_environment() -> str:
    """
    .env dosyasindan GROQ_API_KEY yukler.

    Returns:
        API anahtari string'i.

    Raises:
        SystemExit: API anahtari bulunamazsa programi sonlandirir.
    """
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error(
            "GROQ_API_KEY bulunamadi. Lutfen .env dosyasini kontrol edin."
        )
        print(
            "\n[HATA] GROQ_API_KEY ortam degiskeni tanimli degil!\n"
            "Lutfen proje dizininde .env dosyasi olusturun ve asagidaki satiri ekleyin:\n"
            "  GROQ_API_KEY=your_api_key_here\n"
        )
        sys.exit(1)
    logger.info("GROQ_API_KEY basariyla yuklendi.")
    return api_key


# --------------------------------------------------------------------------- #
#  LLM Baslatma
# --------------------------------------------------------------------------- #

def initialize_llm(api_key: str, model: str = "llama-3.3-70b-versatile") -> ChatGroq:
    """
    ChatGroq LLM nesnesini baslatir.

    Args:
        api_key: Groq API anahtari.
        model: Kullanilacak model adi.

    Returns:
        Baslatilmis ChatGroq nesnesi.

    Raises:
        SystemExit: LLM baslatilmazsa programi sonlandirir.
    """
    try:
        llm = ChatGroq(
            api_key=api_key,
            model_name=model,
            temperature=0.1,
            max_tokens=2048,
        )
        logger.info("LLM baslatildi: model='%s'", model)
        return llm
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM baslatilmadi: %s", exc)
        print(f"\n[HATA] LLM baslatilmadi: {exc}\n")
        sys.exit(1)


# --------------------------------------------------------------------------- #
#  Grafik Olusturma Pipeline
# --------------------------------------------------------------------------- #

def build_knowledge_graph(
    llm: ChatGroq,
    file_path: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> tuple:
    """
    Belge isleme ve graf olusturma pipeline'ini calistirir.

    Args:
        llm: Baslatilmis LLM nesnesi.
        file_path: Islenecek belge yolu.
        chunk_size: Chunk basina kelime sayisi.
        chunk_overlap: Chunk ortusme miktari.

    Returns:
        Tuple[QueryEngine, dict]: QueryEngine ve graf ozet bilgisi.
    """
    # --- Asama 1: Belge isleme ---
    logger.info("=== ASAMA 1: Belge Isleme ===")
    try:
        processor = DocumentProcessor(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = processor.process(file_path)
    except FileNotFoundError as exc:
        logger.error("Belge bulunamadi: %s", exc)
        print(f"\n[HATA] {exc}\n")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        logger.error("Belge isleme hatasi: %s", exc)
        print(f"\n[HATA] Belge islenirken hata olustu: {exc}\n")
        sys.exit(1)

    print(f"\n[OK] Belge {len(chunks)} parcaya bolundu.")

    # --- Asama 2: Graf olusturma ---
    logger.info("=== ASAMA 2: Bilgi Grafigi Olusturma ===")
    print("\n[>>] Bilgi Grafigi olusturuluyor (bu islem birkac dakika surebilir)...")
    print("     Her chunk icin LLM ile varlik/iliski cikarimi yapiliyor...\n")

    try:
        builder = GraphBuilder(llm=llm)
        graph = builder.build_from_chunks(chunks)
        summary = builder.get_graph_summary()
    except Exception as exc:  # noqa: BLE001
        logger.error("Graf olusturma hatasi: %s", exc)
        print(f"\n[HATA] Graf olusturulurken hata: {exc}\n")
        sys.exit(1)

    print(
        f"\n[OK] Bilgi Grafigi hazir: {summary['node_count']} dugum, "
        f"{summary['edge_count']} kenar"
    )

    # --- Asama 3: Sorgu motoru ---
    engine = QueryEngine(llm=llm, graph=graph, hop_depth=2)
    return engine, summary


# --------------------------------------------------------------------------- #
#  Etkilesimli Mod
# --------------------------------------------------------------------------- #

def run_interactive_mode(engine: QueryEngine, summary: dict) -> None:
    """
    Etkilesimli CLI soru-cevap dongusunu baslatir.

    Args:
        engine: Baslatilmis QueryEngine nesnesi.
        summary: Graf ozet bilgisi dict'i.
    """
    logger.info("Etkilesimli mod baslatildi.")
    print("\n" + "=" * 66)
    print("  [GRAPH RAG] Etkilesimli Sorgu Modu")
    print("=" * 66)
    print(f"  Graf: {summary['node_count']} varlik | {summary['edge_count']} iliski")
    print("  Cikmak icin: 'exit', 'quit' veya Ctrl+C yazin")
    print("  Graf istatistikleri icin: 'stats'")
    print("=" * 66 + "\n")

    while True:
        try:
            user_input = input(">> Sorunuz: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGraphRAG kapatiliyor. Iyi calismalar!")
            logger.info("Kullanici etkilesimli modu sonlandirdi.")
            break

        if not user_input:
            print("  [!] Lutfen bir soru girin.\n")
            continue

        if user_input.lower() in {"exit", "quit", "q", "cikis", "cikis"}:
            print("\nGraphRAG kapatiliyor. Iyi calismalar!")
            logger.info("Kullanici 'exit' komutuyla cikti.")
            break

        # Graf istatistikleri komutu
        if user_input.lower() in {"stats", "istatistik", "graf"}:
            stats = engine.get_graph_stats()
            print(
                f"\n[STATS] Graf Istatistikleri:\n"
                f"   Dugum sayisi : {stats['nodes']}\n"
                f"   Kenar sayisi : {stats['edges']}\n"
                f"   Yonlu graf   : {'Evet' if stats['is_directed'] else 'Hayir'}\n"
            )
            continue

        # Sorguyu isle
        print("\n[>>] Bilgi Grafigi uzerinde arastiriliyor...\n")
        logger.info("Kullanici sorgusu: '%s'", user_input)

        try:
            result = engine.answer(user_input)
            print("-" * 66)
            print(f"[YANIT]\n\n{result['answer']}")
            print("-" * 66)
            print(
                f"   [*] Baglamda kullanilan dugum sayisi: {result['context_node_count']}"
            )
            print()
        except Exception as exc:  # noqa: BLE001
            logger.error("Sorgu islenirken hata: %s", exc)
            print(f"\n[HATA] Sorgu islenirken beklenmedik hata: {exc}\n")


# --------------------------------------------------------------------------- #
#  Tek Sorgu Modu
# --------------------------------------------------------------------------- #

def run_single_query(engine: QueryEngine, query: str) -> None:
    """
    Tek bir sorguyu isleyip ciktiyi ekrana yazar.

    Args:
        engine: QueryEngine nesnesi.
        query: Islenecek sorgu metni.
    """
    logger.info("Tek sorgu modu: '%s'", query)
    print(f"\n[SORGU] {query}\n")
    print("[>>] Bilgi Grafigi uzerinde arastiriliyor...\n")

    try:
        result = engine.answer(query)
        print("-" * 66)
        print(f"[YANIT]\n\n{result['answer']}")
        print("-" * 66)
        print(f"\n   [*] Baglamda kullanilan dugum sayisi: {result['context_node_count']}")
    except Exception as exc:  # noqa: BLE001
        logger.error("Tek sorgu islenirken hata: %s", exc)
        print(f"\n[HATA] {exc}\n")


# --------------------------------------------------------------------------- #
#  CLI Arguman Ayristirici
# --------------------------------------------------------------------------- #

def parse_arguments() -> argparse.Namespace:
    """CLI argumanlarini tanimlar ve ayristirir."""
    parser = argparse.ArgumentParser(
        prog="graphrag",
        description=(
            "GraphRAG - Bilgi Grafigi Tabanli Otonom RAG Sistemi\n"
            "LangChain + Groq (Llama-3) + NetworkX ile guclendirilmistir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ornekler:\n"
            "  python main.py --interactive\n"
            "  python main.py --query 'Alan Turing kimdir?'\n"
            "  python main.py --file custom_doc.txt --interactive\n"
        ),
    )
    parser.add_argument(
        "--file",
        type=str,
        default="sample_data.txt",
        metavar="DOSYA",
        help="Islenecek metin dosyasinin yolu (varsayilan: sample_data.txt)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Etkilesimli soru-cevap modunu baslatir",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        metavar="SORU",
        help="Tek bir sorgu calistir ve cik",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.3-70b-versatile",
        metavar="MODEL",
        help="Kullanilacak Groq modeli (varsayilan: llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        metavar="N",
        help="Chunk basina kelime sayisi (varsayilan: 300)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        metavar="N",
        help="Chunk ortusme kelime sayisi (varsayilan: 50)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log seviyesi (varsayilan: INFO)",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  Ana Giris Noktasi
# --------------------------------------------------------------------------- #

def main() -> None:
    """Ana program akisi."""
    args = parse_arguments()
    setup_logging(args.log_level)

    print(BANNER)
    logger.info("GraphRAG baslatildi.")
    logger.info("Parametre: file='%s', model='%s'", args.file, args.model)

    # Ortam ve LLM
    api_key = load_environment()
    llm = initialize_llm(api_key=api_key, model=args.model)

    # Bilgi Grafigi olustur
    engine, summary = build_knowledge_graph(
        llm=llm,
        file_path=args.file,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Sorgu modunu sec
    if args.interactive:
        run_interactive_mode(engine, summary)
    elif args.query:
        run_single_query(engine, args.query)
    else:
        # Hicbir mod secilmemisse etkilesimli modu baslatir
        logger.info("Mod belirtilmedi, etkilesimli mod baslatiliyor.")
        print(
            "\n  [!] --interactive veya --query parametresi belirtilmedi.\n"
            "      Etkilesimli mod otomatik olarak baslatiliyor.\n"
        )
        run_interactive_mode(engine, summary)

    logger.info("GraphRAG oturumu sona erdi.")


if __name__ == "__main__":
    main()

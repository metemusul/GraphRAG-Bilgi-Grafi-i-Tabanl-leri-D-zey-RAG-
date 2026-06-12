"""
document_processor.py
---------------------
Metin belgelerini okuyup anlamlı parçalara (chunk) bölen modül.
Kurumsal düzeyde hata yönetimi ve loglama içerir.
"""

import logging
import os
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Metin belgesini okur ve örtüşen (overlapping) parçalara böler.

    Attributes:
        chunk_size (int): Her bir parçanın yaklaşık kelime sayısı.
        chunk_overlap (int): Parçalar arasındaki örtüşen kelime sayısı.
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size pozitif bir tam sayı olmalıdır.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap, 0 ile chunk_size arasında olmalıdır."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.debug(
            "DocumentProcessor başlatıldı: chunk_size=%d, chunk_overlap=%d",
            chunk_size,
            chunk_overlap,
        )

    def load_document(self, file_path: str) -> str:
        """
        Belirtilen dosya yolundan metni okur.

        Args:
            file_path: Okunacak metin dosyasının yolu.

        Returns:
            Dosyanın tüm içeriği string olarak.

        Raises:
            FileNotFoundError: Dosya bulunamazsa.
            IOError: Dosya okuma hatası durumunda.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("Dosya bulunamadı: %s", file_path)
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

        if not path.is_file():
            logger.error("Belirtilen yol bir dosya değil: %s", file_path)
            raise ValueError(f"Belirtilen yol bir dosya değil: {file_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info(
                "Belge başarıyla yüklendi: '%s' (%d karakter)",
                file_path,
                len(content),
            )
            return content
        except UnicodeDecodeError:
            logger.warning(
                "UTF-8 okuma başarısız, latin-1 deneniyor: %s", file_path
            )
            try:
                with open(path, "r", encoding="latin-1") as f:
                    content = f.read()
                logger.info("Belge latin-1 ile okundu: '%s'", file_path)
                return content
            except IOError as exc:
                logger.error("Dosya okunamadı: %s — %s", file_path, exc)
                raise IOError(f"Dosya okunamadı: {file_path}") from exc

    def chunk_text(self, text: str) -> List[str]:
        """
        Verilen metni örtüşen kelimelere dayalı parçalara böler.

        Args:
            text: Bölünecek ham metin.

        Returns:
            String listesi olarak metin parçaları.
        """
        if not text or not text.strip():
            logger.warning("Boş metin chunk işlemine alındı, boş liste döndürülüyor.")
            return []

        words = text.split()
        total_words = len(words)
        chunks: List[str] = []
        start = 0

        while start < total_words:
            end = min(start + self.chunk_size, total_words)
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            logger.debug(
                "Chunk %d oluşturuldu: kelime %d-%d", len(chunks), start, end
            )
            if end == total_words:
                break
            start += self.chunk_size - self.chunk_overlap

        logger.info(
            "Toplam %d kelimeden %d chunk oluşturuldu.", total_words, len(chunks)
        )
        return chunks

    def process(self, file_path: str) -> List[str]:
        """
        Belgeyi yükler ve chunk'lara böler. Tüm pipeline'ı çalıştırır.

        Args:
            file_path: İşlenecek belge yolu.

        Returns:
            Metin parçalarının listesi.
        """
        logger.info("Belge işleme başlıyor: %s", file_path)
        text = self.load_document(file_path)
        chunks = self.chunk_text(text)
        logger.info("Belge işleme tamamlandı. %d chunk hazır.", len(chunks))
        return chunks

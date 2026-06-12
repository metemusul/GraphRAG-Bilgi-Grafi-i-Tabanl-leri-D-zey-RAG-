# GraphRAG — Bilgi Grafiği Tabanlı Otonom RAG Sistemi

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![Groq](https://img.shields.io/badge/Groq_API-Llama--3-F55036?style=for-the-badge&logo=meta&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-3.3%2B-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**LangChain + Groq (Llama-3) + NetworkX ile güçlendirilmiş, kurumsal seviye GraphRAG CLI uygulaması**

</div>

---

## 🎯 Amaç

Geleneksel RAG (Retrieval-Augmented Generation) sistemleri, metni parçalara bölerek vektör veritabanında saklar ve anlamsal benzerliğe göre bağlam alır. Bu yaklaşım, **varlıklar arası ilişkileri** ve **çok-atlamalı (multi-hop) çıkarımları** yakalamakta yetersiz kalır.

**GraphRAG**, bu sorunu şu şekilde çözer:

1. 📄 Metni anlamlı parçalara böler
2. 🤖 Her parçadan LLM ile **varlıklar** (kişi, kurum, ürün, kavram) ve **ilişkiler** çıkarır
3. 🕸️ Bu varlık-ilişki çiftlerini bir **Bilgi Grafiğine (Knowledge Graph)** dönüştürür
4. 🔍 Kullanıcı sorusu geldiğinde ilgili **alt-grafi (subgraph)** bulur
5. 💡 Bu zengin bağlamı LLM'e vererek **sentezlenmiş, ilişkisel** bir yanıt üretir

---

## 🏗️ Proje Mimarisi

```
graphrag/
├── document_processor.py  # Belge okuma ve chunk'lama
├── graph_builder.py       # LLM çıkarımı + NetworkX DiGraph
├── query_engine.py        # Subgraf keşfi + LLM yanıt üretimi
├── main.py                # CLI arayüzü ve orkestrasyon
├── sample_data.txt        # Test belgesi (Yapay Zeka & Uzay tarihi)
├── requirements.txt       # Python bağımlılıkları
├── .env                   # API anahtarları (git'e eklenmez!)
└── README.md
```

### Varlık-İlişki → Graf Dönüşümü

```
Metin: "IBM, Deep Blue ile Kasparov'u yendi."
           ↓ LLM Çıkarımı
   {
     entities: [IBM, Deep_Blue, Kasparov],
     relations: [IBM --geliştirdi--> Deep_Blue,
                 Deep_Blue --yendi--> Kasparov]
   }
           ↓ NetworkX DiGraph
   [IBM] ──geliştirdi──▶ [Deep Blue]
                              │
                            yendi
                              ▼
                          [Kasparov]
```

### Sorgu Akışı

```
Kullanıcı Sorusu
      ↓
Token Eşleştirme (Graf Düğümleri)
      ↓
BFS/DFS Subgraf Genişletme (2-hop)
      ↓
Bağlam Metni Oluşturma
      ↓
LLM (Llama-3 via Groq) → Sentezlenmiş Yanıt
```

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Amaç |
|---|---|
| **LangChain** | LLM orkestrasyon çerçevesi |
| **langchain-groq** | Groq API entegrasyonu |
| **Groq API** | Ücretsiz, hızlı LLM inference |
| **Llama-3 70B** | Varlık çıkarımı ve yanıt üretimi |
| **NetworkX** | Yönlü Bilgi Grafiği (DiGraph) |
| **python-dotenv** | Güvenli API anahtarı yönetimi |
| **Python logging** | Kurumsal seviye loglama |

---

## ⚡ Kurulum ve Kullanım

### Ön Gereksinimler

- Python 3.10 veya üzeri
- Ücretsiz Groq API anahtarı → [console.groq.com](https://console.groq.com)

### 1. Depoyu Klonla

```bash
git clone https://github.com/kullaniciadi/graphrag.git
cd graphrag
```

### 2. Sanal Ortam Oluştur ve Aktif Et

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 4. API Anahtarını Yapılandır

Proje dizininde `.env` dosyası oluştur:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Uygulamayı Çalıştır

**Etkileşimli mod (önerilen):**
```bash
python main.py --interactive
```

**Tek sorgu modu:**
```bash
python main.py --query "Alan Turing yapay zeka tarihine nasıl katkı sağladı?"
```

**Özel belge ile:**
```bash
python main.py --file belgem.txt --interactive
```

**Tüm parametreler:**
```bash
python main.py --help
```

---

## 🎮 CLI Parametreleri

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--file` | `sample_data.txt` | İşlenecek metin dosyası |
| `--interactive` | `False` | Etkileşimli soru-cevap modu |
| `--query` | `None` | Tek sorgu çalıştır ve çık |
| `--model` | `llama3-70b-8192` | Groq modeli |
| `--chunk-size` | `300` | Chunk başına kelime sayısı |
| `--chunk-overlap` | `50` | Örtüşme kelime sayısı |
| `--log-level` | `INFO` | Log seviyesi |

---

## 💬 Örnek Kullanım

```
❓ Sorunuz: IBM'in yapay zeka tarihindeki rolü nedir?

🔍 Bilgi Grafiği üzerinde araştırılıyor...

──────────────────────────────────────────────────────────────
💡 Yanıt:

IBM, yapay zeka tarihinde kritik bir dönüm noktası oluşturmuştur.
1997 yılında geliştirdiği Deep Blue adlı satranç bilgisayarı,
dönemin Dünya Satranç Şampiyonu Garry Kasparov'u mağlup etmiş ve
yapay zekanın belirli alanlarda insanı aşabileceğini kanıtlamıştır.

Özet: IBM → Deep Blue → Kasparov zinciri, hesaplama zekasının
insan performansını geçtiğini gösteren tarihi ilk kanıttır.
──────────────────────────────────────────────────────────────
   📌 Bağlamda kullanılan düğüm sayısı: 5
```

---

## 📊 Örnek Veri (sample_data.txt)

Proje, yapay zeka tarihi ve uzay araştırmaları hakkında 4 paragraflık
bir örnek belge içerir. İçerikte şu varlıklar yer almaktadır:

- **Kişiler:** Alan Turing, John McCarthy, Garry Kasparov, Elon Musk, Jeff Bezos
- **Kuruluşlar:** IBM, Google, OpenAI, Microsoft, Anthropic, Meta, SpaceX, NASA, ESA
- **Ürünler/Sistemler:** Deep Blue, AlphaGo, ChatGPT, Falcon 9, Starship, James Webb

---

## 🔒 Güvenlik

- API anahtarları yalnızca `.env` dosyasında saklanır
- `.env` dosyası asla Git'e eklenmemelidir
- `.gitignore`'a `.env` eklenmiş olmalıdır

---

## 📝 Loglama

Uygulama, hem konsola hem de `graphrag.log` dosyasına log yazar:

```
2024-01-15 10:30:00 | INFO     | __main__                  | GraphRAG başlatıldı.
2024-01-15 10:30:01 | INFO     | document_processor        | Belge başarıyla yüklendi.
2024-01-15 10:30:05 | INFO     | graph_builder             | Graf oluşturma tamamlandı. 28 düğüm, 35 kenar.
```

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) altında dağıtılmaktadır.

```
MIT License

Copyright (c) 2024 GraphRAG Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">

**GraphRAG** — Bilgiyi ilişkisel düşünceyle sentezle 🧠

</div>

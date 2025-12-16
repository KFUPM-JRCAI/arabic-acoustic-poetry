# Arabic Acoustic Poetry: Poem Meter Classification

Code repository for "Poem Meter Classification of Spoken Arabic Poetry: Integrating High-Resource Systems for a Low-Resource Task"

## Description

This repository contains the implementation code for classifying Arabic poetry meters from recited poetry recordings. We explore two approaches:

1. **End-to-End Classification**: Direct audio-to-meter classification using fine-tuned Wav2Vec2
2. **Transcription-Based Classification**: Integration of ASR (Wav2Vec2) and text-based meter classification systems

The transcription-based approach achieves F1-scores between 80-87% by integrating two high-resource systems to address this low-resource task.

## Dataset Information

### 1. Baseline Dataset
- **Samples**: 3,668 transcribed recordings
- **Duration**: 9.277 hours
- **Coverage**: All 16 classical Arabic meters
- **Quality**: Laboratory-controlled recordings
- **Access**: [HuggingFace Dataset](https://huggingface.co/datasets/KFUPM-JRCAI/arabic-acoustic-poetry)
- **License**: Academic/research use only (NOT for commercial use)
- **Owner**: Dr. Abdul Kareem Saleh Al-Zahrani (akareem@kfupm.edu.sa)


### The 16 Classical Arabic Poetry Meters

Both datasets cover all 16 meters (البحور): الطويل، المديد، البسيط، الوافر، الكامل، الهزج، الرجز، الرمل، السريع، المنسرح، الخفيف، المضارع، المقتضب، المجتث، المتقارب، المتدارك

## Code Information

### Repository Structure

```
ModelingCode/                    
├── end_to_end_BiGRU_model.ipynb
├── FineTuneArabicSpeechCropus.ipynb
├── Wav2vecWithKenLMPatrick.ipynb
├── Wav2vecWithKenLMPatrickFinetunedWav2vec.ipynb
├── evaluate_classification_models.ipynb
├── baseline_test_results.ipynb
└── dataset_explorer.ipynb

FurtherModelingCode/Notebooks&Code/  
├── end_to_end_simple_dense_model.ipynb
├── text_meter_classification_on_our_dataset.ipynb
└── models_evaluation_on_test_sets/
    ├── EndToEnd/
    └── TranscriptionBased/
```

### Notebooks

**End-to-End Models:**
- `end_to_end_BiGRU_model.ipynb` - BiGRU-based classification
- `end_to_end_simple_dense_model.ipynb` - Dense layer classifier

**Transcription-Based Models:**
- `Wav2vecWithKenLMPatrick.ipynb` - Wav2Vec2 + KenLM integration
- `Wav2vecWithKenLMPatrickFinetunedWav2vec.ipynb` - Fine-tuned variant
- `text_meter_classification_on_our_dataset.ipynb` - Text-based meter classification

**Evaluation:**
- `evaluate_classification_models.ipynb` - Model comparison
- `SinAIBaseModelResults.ipynb` / `SinAIFineTunedModelResults.ipynb` - Results analysis

## Usage Instructions

### Prerequisites

**All notebooks were developed and tested on Google Colab.** For best compatibility, use Google Colab with GPU runtime.

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/arabic-acoustic-poetry.git
cd arabic-acoustic-poetry

# Install dependencies (in Colab or local environment)
pip install -r requirements.txt

# KenLM (for language model integration)
pip install https://github.com/kpu/kenlm/archive/master.zip
```

### Loading Datasets

```python
from datasets import load_dataset

# Baseline dataset (research use only)
baseline_dataset = load_dataset("KFUPM-JRCAI/arabic-acoustic-poetry-baseline")

# Benchmark dataset
benchmark_datasest = load_dataset("KFUPM-JRCAI/arabic-acoustic-poetry-benchmark")
```

### Code Execution

1. **Data Exploration**: Start with `dataset_explorer.ipynb`
2. **Training**: e.g. Run `Wav2vecWithKenLMPatrickFinetunedWav2vec.ipynb`
3. **Evaluation**: Use `evaluate_classification_models.ipynb`


## Requirements

**Environment**: Google Colab (all notebooks were run on Google Colab)

**Major Dependencies** (see [requirements.txt](requirements.txt)):
- PyTorch
- Transformers
- Datasets
- Audio: librosa, torchaudio, soundfile
- Arabic NLP: PyArabic, arabic-reshaper, python-bidi
- ML: scikit-learn, numpy, pandas

## Methodology

### End-to-End Architecture
- Based on Wav2Vec2 pretrained model
- Replaces CTC layer with classification head (Dense/BiGRU)
- Fine-tuned for 16-class meter prediction

### Transcription-Based Architecture
1. **Speech Recognition**: Wav2Vec2 (base or fine-tuned) transcribes audio
2. **Language Model**: KenLM 4-gram enhances transcriptions
3. **Meter Classification**: Text classifier (Abandah et al., 2020) identifies meter


## Citation

If you use this code or datasets, please cite:

```bibtex
@article{al2025poem,
  title={Poem Meter Classification of Recited Arabic Poetry: Integrating High-Resource Systems for a Low-Resource Task},
  author={Al-Shaibani, Maged S and Alyafeai, Zaid and Ahmad, Irfan},
  journal={arXiv preprint arXiv:2504.12172},
  year={2025}
}
```

**Paper**: https://arxiv.org/pdf/2504.12172

## License and Usage Terms

### Code
MIT License (or specify your license)

### Baseline Dataset
**IMPORTANT**: Academic/research use ONLY. Commercial use **PROHIBITED** without permission.

**Contact for Commercial Use:**
- Dr. Abdul Kareem Saleh Al-Zahrani
- Email: akareem@kfupm.edu.sa
- Affiliation: Islamic & Arabic Studies Department, KFUPM
- Faculty Page: https://faculty.kfupm.edu.sa/ias/akareem/index.htm

### Benchmark Dataset
Publicly available for research use.

## Resources

- **Datasets**: [HuggingFace Collection](https://huggingface.co/collections/KFUPM-JRCAI/arabic-acoustic-poetry)
- **Paper**: https://arxiv.org/pdf/2504.12172

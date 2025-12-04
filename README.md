# Arabic Acoustic Poetry: Poem Meter Classification

A research project for classifying Arabic poetry meters from acoustic sources by integrating high-resource speech recognition and text-based meter classification systems.

## Overview

This repository contains the implementation for poem meter classification of recited Arabic poetry. The research explores two distinct approaches:

1. **End-to-End Classification**: Direct classification from audio to meter using fine-tuned Wav2Vec2
2. **Transcription-Based Classification**: Integration of speech recognition (ASR) and text-based meter classification systems

Our findings demonstrate that the transcription-based approach, which combines two high-resource systems, effectively addresses this low-resource task, achieving F1-scores between 80% and 87%.

## Features

- Two complementary datasets: baseline training dataset and public benchmark evaluation set
- Integration of Wav2Vec2 for speech recognition with specialized meter classification
- Language model enhancement for improved transcription accuracy
- Support for all 16 classical Arabic poetry meters

## Datasets

### 1. Arabic Acoustic Poetry Dataset (Baseline)

The baseline dataset comprises 3,668 transcribed acoustic recordings collected in controlled laboratory conditions.

**Statistics:**
- **Total Samples**: 3,668 annotated audio files
- **Duration**: 9.277 hours (~9 hours 17 minutes)
- **Average Sample Length**: 8.985 seconds
- **Storage Size**: 897 MB
- **Meters Covered**: All 16 classical Arabic meters
- **Recording Quality**: Laboratory-controlled, professional equipment

**Dataset Structure:**
```python
DatasetDict({
    train: Dataset({
        features: ['Script', 'Bahr', 'Source', 'audio'],
        num_rows: 2934
    })
    validation: Dataset({
        features: ['Script', 'Bahr', 'Source', 'audio'],
        num_rows: 367
    })
    test: Dataset({
        features: ['Script', 'Bahr', 'Source', 'audio'],
        num_rows: 367
    })
})
```

**Features:**
- `Script`: Text transcription of the recited verse (Arabic)
- `Bahr`: Poetry meter label (one of 16 classical meters)
- `Source`: Source information for the audio sample
- `audio`: WAV format audio recording

**Access**: Available on HuggingFace for research purposes.
- Dataset: https://huggingface.co/datasets/KFUPM-JRCAI/arabic-acoustic-poetry

**License**: **NOT FOR COMMERCIAL USE**. This dataset is provided for academic and research purposes only. Commercial use requires explicit permission from the dataset owner.

**Dataset Owner:**
- Dr. Abdul Kareem Saleh Al-Zahrani
- Email: akareem@kfupm.edu.sa
- Faculty Page: https://faculty.kfupm.edu.sa/ias/akareem/index.htm

### 2. Arabic Acoustic Poetry Benchmark

A publicly available evaluation dataset with 268 audio samples designed for standardized model evaluation.

**Statistics:**
- **Total Samples**: 268 annotated files
- **Duration**: 0.48 hours (~29 minutes)
- **Average Sample Length**: 6.442 seconds
- **Storage Size**: 105 MB
- **Distribution**: Balanced across all 16 meters (10-25 samples per meter)

**Access**: Publicly available on HuggingFace
- Dataset: https://huggingface.co/datasets/KFUPM-JRCAI/arabic-acoustic-poetry-benchmark

### The 16 Classical Arabic Poetry Meters

Both datasets cover all 16 meters defined in Arabic prosody ('Arud):

1. الطويل (Al-Taweel)
2. المديد (Al-Madeed)
3. البسيط (Al-Baseet)
4. الوافر (Al-Wafer)
5. الكامل (Al-Kamel)
6. الهزج (Al-Hazaj)
7. الرجز (Al-Rajaz)
8. الرمل (Al-Ramal)
9. السريع (Al-Saree')
10. المنسرح (Al-Munsareh)
11. الخفيف (Al-Khafeef)
12. المضارع (Al-Mudhare')
13. المقتضب (Al-Muqtadhab)
14. المجتث (Al-Mujtath)
15. المتقارب (Al-Mutaqareb)
16. المتدارك (Al-Mutadarek)

## Model Collection

All trained models and systems are available in the HuggingFace collection:

**Collection**: https://huggingface.co/collections/KFUPM-JRCAI/arabic-acoustic-poetry

## Methodology

### End-to-End Architecture

Fine-tuned Wav2Vec2 model with modified classification head:
- Removes CTC layer from pretrained Wav2Vec2
- Adds dense layer + softmax for 16-class meter prediction
- Transfer learning approach for limited data scenarios

### Transcription-Based Architecture

Two-stage pipeline integrating high-resource systems:

1. **Speech Recognition**: Wav2Vec2 transcribes audio to text
   - Base model: Pretrained Arabic Wav2Vec2
   - Fine-tuned variant: Adapted on poetry dataset
   - Language model: 4-gram model trained on Arabic Poetry Comprehensive Dataset

2. **Meter Classification**: Text-based classification system
   - Uses system from Abandah et al. (2020)
   - Covers all 16 meters plus prose category



## Loading the Dataset

```python
from datasets import load_dataset

# Load baseline dataset (for research purposes only - NOT for commercial use)
dataset = load_dataset("KFUPM-JRCAI/arabic-acoustic-poetry")

# Load public benchmark
benchmark = load_dataset("KFUPM-JRCAI/arabic-acoustic-poetry-benchmark")
```


## Citation

If you use this work in your research, please cite:

```bibtex
@article{alshaibani2025poem,
  title={Poem Meter Classification of Recited Arabic Poetry: Integrating High-Resource Systems for a Low-Resource Task},
  author={Al-Shaibani, Maged S. and Alyafeai, Zaid and Ahmad, Irfan and Al-Zahrani, Abdul Kareem Saleh},
  journal={arXiv preprint arXiv:2504.12172},
  year={2025}
}
```

**Paper**: https://arxiv.org/pdf/2504.12172

## License and Usage Terms

### Baseline Dataset License

**IMPORTANT**: The baseline Arabic Acoustic Poetry Dataset is provided for **academic and research purposes ONLY**.

- **Commercial use is STRICTLY PROHIBITED** without explicit permission
- Research use is permitted with proper citation
- For commercial licensing inquiries, contact the dataset owner

**Dataset Owner:**
- **Dr. Abdul Kareem Saleh Al-Zahrani**
- Email: akareem@kfupm.edu.sa
- Affiliation: Islamic & Arabic Studies Department, King Fahd University of Petroleum and Minerals
- Faculty Page: https://faculty.kfupm.edu.sa/ias/akareem/index.htm

Unauthorized commercial use may result in legal consequences.

### Benchmark Dataset

The benchmark evaluation set is publicly available under standard research terms.

## Related Resources

- **Benchmark Dataset**: https://huggingface.co/datasets/KFUPM-JRCAI/arabic-acoustic-poetry-benchmark
- **Model Collection**: https://huggingface.co/collections/KFUPM-JRCAI/arabic-acoustic-poetry
- **MetRec Dataset**: Large-scale textual Arabic poetry dataset with 55,400 verses

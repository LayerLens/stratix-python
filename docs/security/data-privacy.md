# Data Privacy

This guide covers data privacy considerations and best practices when using the Atlas Python SDK to ensure compliance with privacy regulations and protect sensitive information.

## Overview

When using the Atlas Python SDK, you may be handling sensitive data including:

- **AI model outputs** and evaluation results
- **Prompt data** used in evaluations
- **API credentials** and authentication tokens
- **Organizational information** and project data
- **Usage patterns** and performance metrics

Proper data privacy practices are essential for regulatory compliance and maintaining user trust.

## Data Classification

### Understanding Your Data Types

**Public Data** ✅ (No privacy concerns):
- Model names and identifiers
- Benchmark names and types
- General evaluation statistics
- Documentation and configuration

**Internal Data** ⚠️ (Moderate privacy):
- Evaluation results and scores
- Performance metrics
- Usage analytics
- System logs (without sensitive content)

**Confidential Data** 🔒 (High privacy):
- API keys and credentials
- Custom prompts and datasets
- Proprietary model outputs
- Personal identifiable information (PII)

**Restricted Data** 🚫 (Maximum privacy):
- Personal data under GDPR/CCPA
- Financial or healthcare information
- Trade secrets and intellectual property
- Customer data requiring special handling

### Data Classification Example

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

class DataClassification(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

@dataclass
class EvaluationDataMap:
    """Map Atlas data types to privacy classifications"""
    
    model_name: DataClassification = DataClassification.PUBLIC
    benchmark_name: DataClassification = DataClassification.PUBLIC
    evaluation_scores: DataClassification = DataClassification.INTERNAL
    model_outputs: DataClassification = DataClassification.CONFIDENTIAL
    api_credentials: DataClassification = DataClassification.RESTRICTED
    custom_prompts: DataClassification = DataClassification.CONFIDENTIAL

def classify_atlas_data():
    """Example data classification for Atlas SDK usage"""
    data_map = EvaluationDataMap()
    
    print("🔍 Atlas Data Classification:")
    for field_name, field_value in data_map.__dict__.items():
        privacy_level = field_value.value
        print(f"   {field_name}: {privacy_level.upper()}")
    
    return data_map

classify_atlas_data()
```

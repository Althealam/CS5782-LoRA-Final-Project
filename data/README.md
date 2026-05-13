# Data

This project uses GLUE benchmark datasets loaded through the Hugging Face `datasets` library.

The datasets are not stored directly in this repository. They are downloaded automatically at runtime using commands such as:

```python
from datasets import load_dataset
dataset = load_dataset("glue", task_name)
```

Tasks used in this project include:
- MNLI
- MRPC
- QNLI
- QQP
- SST-2
- CoLA

Please make sure your environment has internet access for the first download.

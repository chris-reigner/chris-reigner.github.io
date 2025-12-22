# What are great data ?

- Accuracy: factual correctness and relevance of samples
- diversity: wide range of use cases covering different topics, contexts, text lengths, writing styles
- complexity: include complex, multi-step reasoning problems and challenging tasks
If your number of jobs. Is too low, use open-source instruction dataset. 70B model can be limited to 1k high quality samples , on the other hand, smaller models need more sample as they need to learn a simpler representation

# Creating an instruction dataset

- Data curation:
  - Task specific: get examples of pair dataset
  - Domain specific: gather and validate quality and relevance of data
  - Rule based filtering, length filtering, format checking, and keyword exclusion add to the quality of data saved for fine tuning
- Data deduplication: remove exact match, fuzzy match (for instance using minxish deduplication) or semantically similar data
- Data decontamination: data leakage between train, valves and test sets
- Data quality evaluation: human, LLM as a judge, reward model
- Data exploration: manual, statistical analysis, clustering
- Data generation: generate synthetic data
- Data augmentation: increase quantity and quality of data (from pre-existing data)
  - In depth evolving: enhance complexity of existing instructions
  - In breadth evolving: expand diversity of the dataset
  - Ultra feedback method: instead of modify instructions, if modifies responses

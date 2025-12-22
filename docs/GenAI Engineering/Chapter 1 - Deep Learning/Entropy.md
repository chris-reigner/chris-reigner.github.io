# Understanding Entropy: From Information Theory to Deep Learning

“Knowledge is power. Information is liberating. Education is the premise of progress, in every society, in every family.”, Kofi Annan.
From my overall experience, I always want to understand what the machine (AI) is trying to model and understand.

This page aims at explaining very simply what is entropy and how important it is.

## What is Entropy?

Machine Learning and Deep Learning is about learning, understanding information contained in some data and the ability to generalize this understanding.
**Entropy** is a measure of uncertainty, randomness, or "surprise" in information. Think of it as answering the question: "How much information do I need to describe what's happening?"

The more unpredictable something is, the higher its entropy. The more predictable something is, the lower its entropy.

### 5-Year-Old Example: The Toy Box Game 🧸

Imagine you have a toy box with different colored balls:

**Low Entropy Toy Box:**

- 9 red balls, 1 blue ball
- If I ask you to guess what color I'll pick, you'd almost always say "red" and be right!
- This is predictable = LOW entropy

**High Entropy Toy Box:**

- 5 red balls, 5 blue balls  
- Now guessing is much harder - it's like flipping a coin!
- This is unpredictable = HIGH entropy

---

## Shannon Entropy

Shannon entropy measures the average amount of information needed to describe the outcome of a random event.

**Formula:**

```
H(X) = -Σ p(x) × log₂(p(x))
```

Where:

- H(X) = Shannon entropy
- p(x) = probability of outcome x
- log₂ = logarithm base 2 (measures information in "bits")

### 5-Year-Old Example: Guessing Games 🎯

Let's use our toy box example:

**Low Entropy Box (9 red, 1 blue):**

- Probability of red: 9/10 = 0.9
- Probability of blue: 1/10 = 0.1
- Shannon entropy = -(0.9 × log₂(0.9) + 0.1 × log₂(0.1)) ≈ 0.47 bits

**High Entropy Box (5 red, 5 blue):**

- Probability of red: 5/10 = 0.5  
- Probability of blue: 5/10 = 0.5
- Shannon entropy = -(0.5 × log₂(0.5) + 0.5 × log₂(0.5)) = 1.0 bits

**What this means:** You need more "yes/no questions" on average to figure out what color was picked from the balanced box!

---

## Cross Entropy

**Definition:** Cross entropy measures how well one probability distribution (your guess) predicts another probability distribution (the actual truth).

**Formula:**

```
H(p,q) = -Σ p(x) × log(q(x))
```

Where:

- p(x) = true probability distribution
- q(x) = predicted probability distribution

---

## KL Divergence (Kullback-Leibler Divergence)

**Definition:** KL divergence measures how different one probability distribution is from another. It's like asking: "How much extra information do I need if I use the wrong distribution instead of the right one?"

**Formula:**

```
D_KL(P||Q) = Σ p(x) × log(p(x)/q(x))
```

Where:

- P = true distribution
- Q = approximate distribution

**Key Property:** KL divergence is always ≥ 0, and equals 0 only when P = Q (distributions are identical)

---

## How They All Connect 🔗

Think of it like this:

1. **Shannon Entropy:** "How hard is it to guess what will happen?"
2. **Cross Entropy:** "How surprised am I by what actually happened, given my predictions?"
3. **KL Divergence:** "How much worse is my guessing strategy compared to the perfect strategy?"

**Mathematical Relationship:**

```
Cross Entropy = Shannon Entropy + KL Divergence
H(p,q) = H(p) + D_KL(p||q)
```

---

## Why This Matters in Machine Learning and Deep Learning 🤖

**Shannon entropy** helps understand data complexity and the unpredictability of language itself.
**Cross entropy** is commonly used as a loss function for classification.
**KL divergence** helps in regularization and comparing model outputs.

**The Goal:** Train models to make predictions that minimize cross entropy (be less surprised by the actual answers) and minimize KL divergence (be as close as possible to the true distribution).

| Concept | What it Measures | 5-Year-Old Version |
|---------|------------------|-------------------|
| **Shannon Entropy** | Uncertainty in data | "How hard is the guessing game?" |
| **Cross Entropy** | Prediction quality | "How surprised am I by the real answer?" |
| **KL Divergence** | Difference between distributions | "How much worse is my strategy than the perfect one?" |

---

## KL Divergence in LLMs

**Application 1: Temperature Sampling**

When generating text, LLMs can adjust their "creativity" using temperature:

**Original model output:** "sunny" (60%), "cloudy" (25%), "rainy" (10%), "windy" (5%)

**Low temperature (0.2) - Conservative:**

- "sunny" (80%), "cloudy" (15%), "rainy" (4%), "windy" (1%)
- **KL divergence:** Small (close to original)
- **Result:** More predictable, "safer" text

**High temperature (1.5) - Creative:**

- "sunny" (35%), "cloudy" (30%), "rainy" (20%), "windy" (15%)
- **KL divergence:** Large (very different from original)
- **Result:** More surprising, creative text

**Application 2: Model Alignment & RLHF**

When training models to be helpful and safe:

- **Base Model Distribution:** Might generate toxic or unhelpful content
- **Target Distribution:** Should generate helpful, harmless content

**KL Divergence Constraint:** Keep the aligned model close enough to the base model so it doesn't "forget" how to speak naturally, but far enough to be safe and helpful.

### Real-World LLM Example: Autocomplete

**Scenario:** You're typing an email: "Thank you for your help with the ___"

**What happens inside the LLM:**

1. **Shannon Entropy Analysis:**
   - Model calculates: How predictable is this context?
   - High-frequency patterns → Lower entropy → More confident predictions

2. **Cross Entropy Training:**
   - Model was trained on millions of similar sentences
   - Learned that "project" (30%), "meeting" (20%), "presentation" (15%) are common
   - Training minimized cross entropy on real email data

3. **KL Divergence in Practice:**
   - **Beam Search:** Explores multiple completions, uses KL divergence to balance between probable and diverse options
   - **Safety Filtering:** Ensures suggestions don't diverge too much from appropriate business language

### Why This Matters for LLM Performance

**Better Entropy Understanding = Better Models:**

1. **Training Efficiency:** Cross entropy loss guides the model toward better predictions
2. **Text Generation Quality:** Entropy measures help balance creativity vs. coherence  
3. **Model Evaluation:** Perplexity (related to cross entropy) measures how "surprised" the model is by test data
4. **Safety & Alignment:** KL divergence helps keep models helpful while maintaining capabilities

### Perplexity: The LLM Metric

Perplexity formula:

```
perplexity = exp( H(p,q) )
```

**What it means:** "On average, how many reasonable word choices does the model think there are?"

**Example:**

- Cross entropy = 1.0 → Perplexity = 2 → "Like choosing between 2 equally good options"
- Cross entropy = 3.0 → Perplexity = 8 → "Like choosing among 8 reasonable options"

**Better models have lower perplexity** = they're less "confused" by language!
Perplexity, then, is essentially a measure of how many options the model finds plausible on average, with lower values indicating fewer options (more confident predictions) and higher values indicating more options (greater uncertainty).

**Advantage**: One of the biggest advantages of perplexity is that it is highly intuitive and explainable in a field that is notoriously opaque

**Inconvenient**: The most important limitation of perplexity is that it does not convey a model’s “understanding.” Perplexity is strictly a measure of uncertainty, and a model being uncertain doesn’t mean it is right or wrong. A model may be correct but unconfident or wrong but confident. So, a perplexity score isn’t a measure of accuracy, just of confidence.

## Python Code snippet

Here's a simple python calculation using pytorch:

```
import torch

def calculate_perplexity(logits, target):
    """
    Calculate perplexity from logits and target labels.

    Args:
    - logits (torch.Tensor): Logits output from the model (batch_size, seq_length, vocab_size).
    - target (torch.Tensor): Ground truth labels (batch_size, seq_length).

    Returns:
    - perplexity (float): The perplexity score.
    """

    # Convert logits to log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Gather the log probabilities for the correct target tokens
    # log_probs has shape (batch_size, seq_length, vocab_size)
    # target has shape (batch_size, seq_length)
    # The gather method will pick the log probabilities of the true target tokens
    target_log_probs = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

    # Calculate the negative log likelihood
    negative_log_likelihood = -target_log_probs

    # Calculate the mean negative log likelihood over all tokens
    mean_nll = negative_log_likelihood.mean()

    # Calculate perplexity as exp(mean negative log likelihood)
    perplexity = torch.exp(mean_nll)

    return perplexity.item()

# Example usage
# Simulate a batch of logits (batch_size=2, seq_length=4, vocab_size=10)
logits = torch.randn(2, 4, 10)
# Simulate ground truth target tokens
target = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

# Calculate perplexity
perplexity = calculate_perplexity(logits, target)
print(f'Perplexity: {perplexity}')
```

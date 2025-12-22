# Numerical representations

Numerical representation plays a crucial role in ML and DL.
Even with bigger model, we're always trying to us the most optimized consumption and this goes through how we represent our vectors.
A big point of optimisation is reducing the size but also the precision of the vectors (sometimes for different purpose like training or inference).

You've got a brief overview of numerical representations

A floating-point number, crucial for representing real numbers in computers, is typically composed of three parts: the sign bit, the exponent, and the mantissa. Understanding these components is key because different applications and hardware platforms utilize various floating-point formats (like FP32, FP16, BF16, etc.), each with distinct characteristics regarding range, precision, and computational cost. This knowledge is vital for optimizing performance and ensuring numerical stability, especially in fields like deep learning and scientific computing.

These three components work together to represent a wide range of real numbers by expressing a number in scientific notation (mantissa * base^exponent), with the sign bit indicating its polarity.

Loading a model using the format it was intended for is very important as not doing it results in loss of quality

### FP64 (Double-Precision Floating-Point)

* **Full Name:** Double-Precision Floating-Point
* **Total Bits:** 64
* **Sign Bits:** 1
* **Exponent Bits:** 11
* **Mantissa Bits:** 52
* **Characteristics:**
  * **Range:** Widest. Can represent a vast range of numbers, from very small to very large.
  * **Precision:** Highest. Offers a high degree of accuracy for calculations.
  * **Trade-offs:** Consumes more memory and bandwidth compared to lower precision formats. Computations can be slower on hardware not optimized for FP64.
* **Common Use Cases/Hardware:** Scientific computing, numerical analysis, financial modeling, and other applications requiring high accuracy. It's the default floating-point type in NumPy and often used in CPU-based computations.

### FP32 (Single-Precision Floating-Point)

* **Full Name:** Single-Precision Floating-Point
* **Total Bits:** 32
* **Sign Bits:** 1
* **Exponent Bits:** 8
* **Mantissa Bits:** 23
* **Characteristics:**
  * **Range:** Wide. Offers a good balance for general-purpose computations.
  * **Precision:** High. Sufficient for many applications, including graphics and standard deep learning model parameters.
  * **Trade-offs:** Standard performance and memory usage.
* **Common Use Cases/Hardware:** Widely used as a standard for many applications, including traditional computer graphics, physics simulations, and as a common format for storing weights and activations in deep learning models. General-purpose GPU computing.

### FP16 (Half-Precision Floating-Point)

* **Full Name:** Half-Precision Floating-Point
* **Total Bits:** 16
* **Sign Bits:** 1
* **Exponent Bits:** 5
* **Mantissa Bits:** 10
* **Characteristics:**
  * **Range:** Limited. More susceptible to overflow (number too large) and underflow (number too small) compared to FP32.
  * **Precision:** Limited. Can lead to loss of information for very fine details.
  * **Trade-offs:** Significantly reduces memory footprint and bandwidth requirements (half of FP32). Can offer substantial speedups on compatible hardware (e.g., NVIDIA Tensor Cores, mobile GPUs) due to higher throughput. Requires careful handling, often using techniques like loss scaling in deep learning to prevent numerical instability.
* **Common Use Cases/Hardware:** Deep learning training and inference, especially on modern GPUs (like NVIDIA with Tensor Cores) and mobile devices where memory and power efficiency are critical.

### BF16 (BFloat16 Floating-Point)

* **Full Name:** Brain Floating-Point Format (BFloat16)
* **Total Bits:** 16
* **Sign Bits:** 1
* **Exponent Bits:** 8 (same as FP32)
* **Mantissa Bits:** 7
* **Characteristics:**
  * **Range:** Wide, comparable to FP32. This makes it more resilient to overflow and underflow issues than FP16, especially during deep learning training.
  * **Precision:** Low. With only 7 mantissa bits, the precision is significantly reduced compared to FP16 and FP32. This can affect tasks requiring fine-grained details but is often acceptable for deep learning models.
  * **Trade-offs:** Offers memory savings similar to FP16 and can provide performance benefits. Its FP32-like range simplifies conversion from FP32 models.
* **Common Use Cases/Hardware:** Primarily used for deep learning training and inference, especially on Google TPUs (Tensor Processing Units) and increasingly supported by newer CPUs and GPUs from NVIDIA and Intel.

### TF32 (TensorFloat-32)

* **Full Name:** TensorFloat-32
* **Total Bits:** TF32 is not a storage format in the same way as FP32 or FP16. For specific operations like matrix multiplications and convolutions on compatible hardware, inputs (which are typically FP32) are internally processed using a 19-bit representation.
* **Sign Bits:** 1 (for internal computation)
* **Exponent Bits:** 8 (for internal computation, same as FP32)
* **Mantissa Bits:** 10 (for internal computation, same as FP16)
* **Characteristics:**
  * **Range:** Wide, same as FP32.
  * **Precision:** Limited, effectively the precision of FP16 for the operations it accelerates.
  * **Trade-offs:** Aims to provide a significant speedup over FP32 for deep learning matrix operations with minimal to no code changes and often without a noticeable loss in accuracy for many workloads. Input and output data for TF32-accelerated operations remain in FP32.
* **Common Use Cases/Hardware:** Accelerating deep learning training and inference computations (specifically matrix math) on NVIDIA Ampere architecture GPUs (e.g., A100) and newer. It's often enabled by default in deep learning libraries like PyTorch and TensorFlow for compatible operations on supported hardware.

### Summary of Floating-Point Formats

| Feature         | FP64    | FP32    | FP16    | BF16    | TF32 (for matmul) |
|-----------------|---------|---------|---------|---------|-------------------|
| Total Bits      | 64      | 32      | 16      | 16      | N/A (19-bit compute path) |
| Sign Bits       | 1       | 1       | 1       | 1       | 1                 |
| Exponent Bits   | 11      | 8       | 5       | 8       | 8                 |
| Mantissa Bits   | 52      | 23      | 10      | 7       | 10                |
| Range           | Widest  | Wide    | Limited | Wide    | Wide              |
| Precision       | Highest | High    | Limited | Low     | Limited           |
| Primary Use     | SciComp | General | DL Speed| DL TPU  | DL Nvidia Speedup |

## Mixed Precision Training

Mixed precision training is a technique used to accelerate the training of deep learning models by using a combination of lower-precision floating-point formats (like FP16 or BF16) and higher-precision formats (like FP32).

### Advantages

Employing mixed precision training offers several key benefits:

* **Speed:** Computations, particularly matrix multiplications and convolutions, can be significantly faster. This is due to specialized hardware support (e.g., Tensor Cores in NVIDIA GPUs) for lower-precision types, which can execute these operations at a much higher throughput than FP32. Peak FP16/BF16 performance can be several times higher than FP32 on modern accelerators. Additionally, reduced data movement (less data to read and write) contributes to speedups.
* **Memory Reduction:** Using lower-precision types like FP16 or BF16 halves the memory footprint for those values compared to FP32. This allows for training larger models (more parameters), using larger batch sizes (which can improve training stability and speed), or processing larger input sizes (e.g., higher resolution images).

### How it Generally Works (Conceptual)

Mixed precision training isn't simply about casting all model parameters and computations to a lower-precision format, as this can lead to numerical instability and accuracy loss. Instead, it involves a more nuanced approach:

* **Master Weights in FP32:** A master copy of the model weights is typically kept in FP32. This ensures that weight updates, which are often small, are accumulated with high precision, preventing loss of information.
* **FP16/BF16 Computations:** During the forward and backward passes, weights and activations are cast to FP16 or BF16 for computation. This is where the speed and memory benefits are realized.
* **Loss Scaling (Primarily for FP16):** FP16 has a more limited dynamic range compared to FP32. This means that very small gradient values (common in deep learning) might become zero in FP16 (a phenomenon called underflow), hindering learning. To counteract this, loss scaling is employed. The loss value is multiplied by a scaling factor before backpropagation, which scales up the gradients. Before the weight update, the gradients are scaled back down to their original magnitude. This helps preserve small gradient values that would otherwise be lost. BF16, having a similar dynamic range to FP32, typically does not require loss scaling.

### Practical Implementation Example

Modern deep learning libraries have significantly simplified the implementation of mixed precision training through Automatic Mixed Precision (AMP) utilities.

* Frameworks like **PyTorch (with `torch.cuda.amp`)** and **TensorFlow (with `tf.keras.mixed_precision`)** provide AMP features. These tools automatically manage the casting of operations to appropriate data types (FP32, FP16, or BF16 based on the operation and hardware) and handle loss scaling where necessary (primarily for FP16). This allows developers to enable mixed precision with minimal code changes, often just a few lines.

### Further Reading

For a detailed guide on mixed precision in PyTorch, see [What Every User Should Know About Mixed Precision Training in PyTorch](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/).

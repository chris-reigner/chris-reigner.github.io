
So far, everything we’ve looked has been focused on text and sequence prediction with language models, but many other “generative AI” techniques require learning distributions with less of a sequential structure (e.g. images). Here we’ll examine a number of non-Transformer architectures for generative modeling, starting from simple mixture models and culminating with diffusion.

# Distribution Modeling

Recalling our first glimpse of language models as simple bigram distributions, the most basic thing you can do in distributional modeling is just count co-occurrence probabilities in your dataset and repeat them as ground truth. This idea can be extended to conditional sampling or classification as “Naive Bayes” (blog post video), often one of the simplest algorithms covered in introductory machine learning courses.

The next generative model students are often taught is the Gaussian Mixture Model and its Expectation-Maximization algorithm; Gaussian Mixture Models + Expectation-Maximization algorithm. This blog post and this video give decent overviews; the core idea here is assuming that data distributions can be approximated as a mixture of multivariate Gaussian distributions. GMMs can also be used for clustering if individual groups can be assumed to be approximately Gaussian.

While these methods aren’t very effective at representing complex structures like images or language, related ideas will appear as components of some of the more advanced methods we’ll see.
<https://mpatacchiola.github.io/blog/2020/07/31/gaussian-mixture-models.html>

# Variational Auto-Encoders

Auto-encoders and variational auto-encoders are widely used for learning compressed representations of data distributions, and can also be useful for “denoising” inputs, which will come into play when we discuss diffusion. Some nice resources:

“Autoencoders” chapter in the “Deep Learning” book
blog post from Lilian Weng
video from Arxiv Insights
blog post from Prakash Pandey on both VAEs and GANs

# Generative Adversarial Nets

The basic idea behind Generative Adversarial Networks (GANs) is to simulate a “game” between two neural nets — the Generator wants to create samples which are indistinguishable from real data by the Discriminator, who wants to identify the generated samples, and both nets are trained continuously until an equilibrium (or desired sample quality) is reached. Following from von Neumann’s minimax theorem for zero-sum games, you basically get a “theorem” promising that GANs succeed at learning distributions, if you assume that gradient descent finds global minimizers and allow both networks to grow arbitrarily large. Granted, neither of these are literally true in practice, but GANs do tend to be quite effective (although they’ve fallen out of favor somewhat in recent years, partly due to the instabilities of simultaneous training).

Resources:

“Complete Guide to Generative Adversarial Networks” from Paperspace
“Generative Adversarial Networks (GANs): End-to-End Introduction” by
Deep Learning, Ch. 20 - Generative Models (theory-focused)

# Conditional GANs

Conditional GANs are where we’ll start going from vanilla “distribution learning” to something which more closely resembles interactive generative tools like DALL-E and Midjourney, incorporating text-image multimodality. A key idea is to learn “representations” (in the sense of text embeddings or autoencoders) which are more abstract and can be applied to either text or image inputs. For example, you could imagine training a vanilla GAN on (image, caption) pairs by embedding the text and concatenating it with an image, which could then learn this joint distribution over images and captions. Note that this implicitly involves learning conditional distributions if part of the input (image or caption) is fixed, and this can be extended to enable automatic captioning (given an image) or image generation (given a caption). There a number of variants on this setup with differing bells and whistles. The VQGAN+CLIP architecture is worth knowing about, as it was a major popular source of early “AI art” generated from input text.

Resources:

“Implementing Conditional Generative Adversarial Networks” blog from Paperspace
“Conditional Generative Adversarial Network — How to Gain Control Over GAN Outputs” by Saul Dobilas
“The Illustrated VQGAN” by LJ Miranda
“Using Deep Learning to Generate Artwork with VQGAN-CLIP” talk from Paperspace

# Diffusion Models

One of the central ideas behind diffusion models (like StableDiffusion) is iterative guided application of denoising operations, refining random noise into something that increasingly resembles an image. Diffusion originates from the worlds of stochastic differential equations and statistical physics — relating to the “Schrodinger bridge” problem and optimal transport for probability distributions — and a fair amount of math is basically unavoidable if you want to understand the whole picture. For a relatively soft introduction, see “A friendly Introduction to Denoising Diffusion Probabilistic Models” by Antony Gitau. If you’re up for some more math, check out “What are Diffusion Models?” for more of a deep dive. If you’re more interested in code and pictures (but still some math), see “The Annotated Diffusion Model”

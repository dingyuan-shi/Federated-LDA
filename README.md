# LDA_Gibbs_sampling_privacy

Source code of paper Federated Latent Dirichlet Allocation: A Local Differential Privacy Based Framework (AAAI 2020)

This program implements a **latent Dirichlet allocation(LDA)** model training by **Gibbs sampling** and **LightLDA(MH + alias table)**.

Also implements privacy version: **local differential privacy(LDP)** and **centralized differential privacy(CDP)**

The Gibbs version is implement by myself, the LightLDA version is based on [this](https://github.com/nzw0301/lightLDA)

The privacy version is implement based on the idea of k-RR, but we add priority.

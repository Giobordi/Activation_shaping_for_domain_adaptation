## DOMAIN ADAPTATION VIA ACTIVATION SHAPING

One of the biggest limitations of deep models is their inability to work consistently well with
any data met at test time. The training and the test set could be characterized by different
data distributions (illumination conditions, visual style, background, etc.) leading to an
inevitable decrease in the model performances. This is known as the problem of domain
shift and it’s one of the most investigated topics in the computer vision community. \  It can be
formalized with the setting of Unsupervised Domain Adaptation (UDA) where there is a
labeled Source domain (Training Set) and an unlabeled Target domain (Test Set), the goal is
to train a model that works well on the target distribution

![immagine](https://github.com/Giobordi/Activation_shaping_for_domain_adaptation/assets/129875197/5f9e7749-353b-4efc-b84e-325c8c85954a)

Activation Shaping is the process of modifying activation maps from a layer within an
architecture using either some user-defined or learned rules and is highly effective for detecting out-of-distribution (OOD) data.In this
project, we aim to leverage the benefits of this OOD detection strategy to exploit the insights
regarding data distribution discrepancies with the goal of learning a model that is robust to
domain shift.


Paper :https://arxiv.org/pdf/1505.07818.pdf
### Domain-adversarial training of neural networks 
Theory in domain adaptation suggest that for an effective domain transfer, the predictions must be made based on features that cannot discriminate between the two domains. \ 
Train the model using the labeled source data and the unlabeled target data, focusing on the features discriminative for the main learning task on the source domain, but not discriminative for the domain shift. This can be achieved in any feed-forward model augmented with a few std layers and a new *gradient reversal layer* (GRL) (that multiplies the gradient by a negative constant during backpropagation). 

The appeal of the Domain adaptation approach is the ability to learn mappings between domains without requiring any target domain labels (unsupervised domain annotation).

The goal is to have a final classification decions based on features that are discriminative and invariant on the change of domain, basing on the idea that a good rappresentation for cross-domain transfer is one for which cannot be distinguished the domain of origin.

The feed-forward model take the name 'domain adversarial neural network' (DANN), use std layers and loss functions and can be trained with standard backpropagation and stochastic gradient descent.
The only non-standard layer is the gradient reversal layer, that leaves the forward propagation unchanged, but multiplies the gradient by a negative constant during backpropagation.  

This approach attempts to  match the feature space distributions modifying the feature space representation rather than by reweighting or geometric transformation, this operation is done training a sequence of deep autoencoders, replacing  the sourrce domain samples with their autoencoder reconstruction (that are expected to be more similar to the target domain samples). Our approach performs feature learning, domain adaptation and classifier learning jointly in a single architecture, using an unsupervised domain adaptation approach.

**Domain adaptation** 

An unsupervised domain adaptation algorithm is probided with a labeled source sample $X=(x_s,y_s)$ from the source domain S and an unlabeled target sample $X=(x_t)$ from the target domain T. The target error is defined by the sum of the source error and the distance between the source and target distributions (many distance measures are available, here the H-divergence). 

*Proxy distance* : it is possible to approximate the H-divergence running a learning algorithm on the problem of discriminating between the source and target samples, so a new dataset is created with the source and target samples and a new label is assigned to each sample, 0 for the source samples and 1 for the target samples.\
Given a generalization error $\epsilon$ the H-divergence is approximated by $\hat{d}_A = 2(1-2\epsilon)$ (colled Proxy A-distance) and the value of $\epsilon$ can be calculated by the classifier error on the new dataset (using SVM or NN).\


**Domain Adversarial Neural Networks (DANN)**
The goal is to learn a model that can generalize well from one domain to another and that the internal representation contains no discriminative information about the origin of the input, preserving a low risk on the source (labeled) example.\
The DANN model is composed by several components : a D-dimensional feature extractor $G_f(*, \theta_f)$, a label prediction output layer $G_y(*, \theta_y)$ and a domain prediction output layer $G_d(*, \theta_d)$, where $\theta_f, \theta_y, \theta_d$ are the parameters.

The prediction loss and the domain loss : 
$$L_{pred}= L_y(G_y(G_f(x_i, \theta_f), \theta_y), y_i)$$ 
$$L_{dom}= L_d(G_d(G_f(x_i, \theta_f), \theta_d), d_i)$$


![immagine](https://github.com/Giobordi/Activation_shaping_for_domain_adaptation/assets/129875197/8cd556a6-69cc-4879-b7dd-f13fb1adf114)

The approach is similar to use a Stochastic Gradient Descent (SGD) , the only difference is that the gradient from the class and the domain prodictor are subtracted, otherwise the SGD would try to make the features dissimilar across domains to minimize the domain classification loss. This is done by adding a gradient reversal layer (GRL), this layer has no parameters so does not require params update, during the forward propagation it acts as an identity function, but during the backpropagation it multiplies the gradient by a negative constant (usually = -1). This new layer is inserted between the feature extractor and the domain classifier.
This approach leads to the emergence of features that are domain-invariant and discriminative at the same time.


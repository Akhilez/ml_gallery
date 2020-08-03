## Selective Generator

The goal of this project is to merge GAN and classifier to generate selected class of images.
For example, using this model, one can generate a MNIST-like handwritten digit from 0-9.

### Structure:

        noise + one-hot-class
                |
            Generator
                |
              Image
                |
               CNNs
       _________|________
       |                |
      FCCs             FCCs
       |                |
    Softmax           Logistic
    Classifier        Discriminator
    
    
```yaml

Generator:
  - Input: 100 (90 noise, 10 one-hot)
  - h1: 150
  - h2: 400
  - h3: 784

FeatureExtractor:
  - Input: 784 (28x28)
  - h1: 400
  - h2: 150
  - h3: 100

Discriminator:
  - Input: 100
  - h1: 50
  - h2: 1

Classifier:
  - Input: 100
  - h1: 50
  - h2: 10

```
    
The model will be created using pytorch's tensors due to the benefit of automatic-gradients.

I wouldn't be using high level nn apis because I want to learn the computations on a low level.

The plan is to implement this model into a web application.
Check the website https://ml.akhilez.com for the main ML website. Main site: https://akhilez.com

To deploy this model into a web application, I plan to use the learned weights as JSON in a tensorflow.js model.


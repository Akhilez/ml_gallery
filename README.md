![MLGalleryLogo](ml_js/src/landing/ml_logo/ml_logo.png)

**Machine Learning Gallery** is a master project of few of my experiments with Neural Networks.
It is designed in a way to help a beginner understand the concepts with visualizations.
You can train and run the networks live and see the results for yourself.

Every project here is followed by an explanation on how it works.
Most models are trained with PyTorch on a Django backend server.
The front-end is a React app which connects to the backend using Websocket.
Some larger models are pre-trained.

Technologies used: __PyTorch, React, TensorFlow JS__

Deployed at: https://akhil.ai


### Intended Projects:

 - Feed-Forward Networks
   - Learn a Line
   - Linear Classifier
   - Learn a Curve (Polynomial)
   - Deep Iris
 - Computer Vision
   - Which Character?
   - MNSIT GAN
   - Colorizer
   - Find The Number
   - Find All Numbers: V1 (Faster-RCNN)
   - Find All Numbers: V2 (Own)
   - Attention, Attention!
   - Style, Please: V1 (Style Transfer)
   - Style, Please: V2 (Style GAN)
 - Natural Language Processing
   - Next Char
   - Word To Vector: V1 (word2vec)
   - Next Word
   - What Genre?
   - Word To Vector: V2 (BERT)
   - Next Sentence
 - Reinforcement Learning
   - TicTacToe
   - Ping-Pong
   - Racer
 - Unsupervised Learning
   - AutoEncoder: V1
   - Self-Organizing Feature Maps
   - Memorize Please (Associative)
 - Miscellaneous
   - Spiking Neurons
   - MNIST Detection Dataset

---

##### API Docs:

A generic flow of control from ui to django:

- api entrypoint => `/api/<project_id>/`
- All actions are post requests with json body

- Page loaded:
  - Request:
    ```json5
    {
      action: 'pre-init',
      data: { /* ... */ }
    }
    ```
- Action button clicked:
    - Request
      ```json5
      {
        job_id: 'uuid',  // Will not exist if initializing job
        action: 'action_key',
        new_job: true,  // If this is the first time calling. (No job_id at client.)
        data: { /* ... */ }
      }
      ```
    - Response
      ```json5
      {
        job_id:  'uuid',  // Store this in client if it doesn't have job_id
        action: 'action_key',
        data: { /* ... */ }
      }
      ```

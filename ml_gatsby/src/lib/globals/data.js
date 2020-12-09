export const DEBUG = process.env.NODE_ENV !== "production"

export const MLPyHost = DEBUG
  ? "http://localhost:8001"
  : "https://py.ml.akhil.ai"
export const HOST = DEBUG ? "http://localhost:8000" : "https://akhil.ai"

export const urls = {
  profile: "https://akhil.ai",
  gallery: "/gallery",
  repo: "https://github.com/Akhilez/ml_gallery",
}

const projectsRaw = {
  learn_line: {
    title: "Learn a Line",
    desc:
      "Visualize the learning of a basic neural network by learning a straight line equation.",
    image: "learn_line.jpg",
    status: "in_progress",
    links: {
      app: "/gallery/linear/learn_line",
      source: "https://github.com/Akhilez/ml_gallery",
    },
  },
  linear_classifier: {
    title: "Linear Classifier",
    desc:
      "Visualize the learning of a linear classifier: learns to distinguish between two different type of points in space that are linearly separable.",
    image: "linear_classifier.png",
    status: "in_progress",
    links: {
      app: "/gallery/linear/linear_classifier",
      source: "https://github.com/Akhilez/ml_gallery",
    },
  },
  learn_curve: {
    title: "Learn A Curve!",
    desc:
      "Neural Networks can learn any continuous function! Here's a model that visualizes this concept. You can add data points and the network will learn its mathematical function.",
    image: "learn_curve.png",
    status: "in_progress",
    links: {
      app: "/gallery/linear/learn_curve",
      source: "https://github.com/Akhilez/ml_gallery",
    },
  },
  deep_iris: {
    title: "Deep Iris",
    desc:
      "Visualize how a multi-layer neural network tries to classify flowers from its petal and sepal dimensions",
    image: "deep_iris.png",
    status: "to_do",
    links: {
      app: "/gallery/linear/deep_iris",
      source: "https://github.com/Akhilez/ml_gallery",
    },
  },

  // vision
  which_char: {
    title: "Which Character?",
    desc: "Draw a number in the box and recognize what number it is",
    image: "which_char.png",
    status: "to_do",
    links: {
      app: "/gallery/vision/which_char",
      source: "https://github.com/Akhilez/ml_gallery",
    },
  },
  mnist_gan: {
    title: "MNIST GAN",
    desc:
      "Generate handwritten numbers using Generative Adversarial Network fused with a Classifier.",
    image: "mnist_gan.png",
    status: "to_do",
    links: {
      app: "/gallery/vision/mnist_gan",
      source: "/mnist_gan",
      colab:
        "https://colab.research.google.com/github/Akhilez/ml_gallery/blob/master/ml_py/app/vision/selective_generator/colab.ipynb",
    },
  },
  colorizer: {
    title: "Colorizer",
    desc:
      "A web application that colorizes grayscale images using a Convolutional Neural Network. This was my major project in my undergrad.",
    image: "colorizer.jpg",
    status: "to_do",
    links: {
      app: "https://akhilez.com/colorizer/",
      source: "https://github.com/Akhilez/ml_gallery",
    },
  },
  find_char: {
    title: "Find The Number",
    desc: "Draw a number in the box and find its location",
    image: "find_number.png",
    status: "to_do",
    links: {
      app: "/gallery/vision/find_char",
      source:
        "https://colab.research.google.com/github/Akhilez/ml_gallery/blob/master/ml_py/app/vision/detector_v1/colab_localization.ipynb",
      colab:
        "https://colab.research.google.com/github/Akhilez/ml_gallery/blob/master/ml_py/app/vision/detector_v1/colab_localization.ipynb",
    },
  },
  positional_cnn: {
    title: "Positional CNN",
    desc: "A CNN architecture that is not positionally invariant",
    image: "under_construction2.png",
    status: "to_do",
    links: {
      app: "/gallery/vision/positional_cnn",
      source:
        "https://github.com/Akhilez/ml_gallery/blob/master/ml_py/app/vision/positional_mnist/positional_mnist.ipynb",
      colab:
        "https://colab.research.google.com/github/Akhilez/ml_gallery/blob/master/ml_py/app/vision/positional_mnist/positional_mnist.ipynb",
    },
  },
  dense_cap: {
    title: "Dense Cap",
    desc: "Generate English captions for numbers and their clusters",
    image: "under_construction2.png",
    status: "to_do",
    links: {
      app: "/gallery/vision/dense_cap",
    },
  },
  find_all_chars: {
    title: "Find All Numbers: V1",
    desc: "",
    image: "faster_rcnn.png",
    status: "to_do",
    links: {
      app: "/gallery/vision/find_all_chars",
      source:
        "https://colab.research.google.com/github/Akhilez/ml_gallery/blob/master/ml_py/app/vision/detector_v1/colab_detection.ipynb",
      colab:
        "https://colab.research.google.com/github/Akhilez/ml_gallery/blob/master/ml_py/app/vision/detector_v1/colab_detection.ipynb",
    },
  },
  find_all_chars_v2: {
    title: "Find All Numbers: V2",
    desc: "",
    image: "faster_rcnn.png",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  image_attention: {
    title: "Attention, Attention!",
    desc: "",
    image: "under_construction2.png",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  style_transfer: {
    title: "Style, Please",
    desc: "",
    image: "under_construction2.png",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  style_gan: {
    title: "Style, Please V2",
    desc: "Style GAN",
    image: "under_construction2.png",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },

  // NLP
  next_char: {
    title: "Next Char",
    desc:
      "Visualize how a Recurrent Neural Network predicts which letter comes next.",
    image: "next_char.png",
    status: "to_do",
    links: {
      app: "/gallery/nlp/next_char",
      source:
        "https://github.com/Akhilez/ml_gallery/blob/master/ml_py/app/nlp/next_char/next_char.py",
      colab:
        "https://colab.research.google.com/github/Akhilez/ml_gallery/blob/master/ml_py/app/nlp/next_char/next_char_colab.ipynb",
    },
  },
  word2vec: {
    title: "Word To Vector",
    desc: "",
    image: "under_construction2.png",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  next_word: {
    title: "Next Word",
    desc: "",
    image: "under_construction2.png",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  word_attention: {
    title: "Word Attention",
    desc: "",
    image: "under_construction2.png",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  word2vec_v2: {
    title: "Word To Vector V2",
    desc: "Using bert",
    image: "under_construction2.png",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  next_sentence: {
    title: "Next Sentence",
    desc: "",
    image: "under_construction2.png",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },

  // Reinforcement Learning
  tictactoe: {
    title: "TicTacToe",
    desc: "",
    image: "tictactoe.jpg",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  ping_ping: {
    title: "Ping-Pong",
    desc: "",
    image: "pong.jpg",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  gridworld: {
    title: "Grid World",
    desc: "Navigate the player to a desired location.",
    image: "gridworld.png",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  racer: {
    title: "Racer",
    desc: "",
    image: "under_construction2.png",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  dodger: {
    title: "Dodger",
    desc: "",
    image: "under_construction2.png",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },

  // Unsupervised
  auto_encoder: {
    title: "AutoEncoder",
    desc: "",
    image: "under_construction2.png",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  self_organizing_map: {
    title: "Self-Organizing Feature Maps",
    desc: "",
    image: "under_construction2.png",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  associative_memory: {
    title: "Memorize Please",
    desc: "",
    image: "under_construction2.png",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },

  // Misc
  spiking_neurons: {
    title: "Spiking Neurons",
    desc: "",
    image: "under_construction2.png",
    status: "to_do",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  mnist_detection_dataset: {
    title: "MNIST Detection Dataset",
    desc: "",
    image: "under_construction2.png",
    status: "in_progress",
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
}

Object.keys(projectsRaw).map(key => {
  projectsRaw[key]["id"] = key
})

export const projects = projectsRaw

export const categoriesMap = {
  linear: {
    title: "Linear Neural Networks",
    projects: [
      projects.learn_line,
      projects.linear_classifier,
      projects.learn_curve,
      projects.deep_iris,
    ],
  },
  vision: {
    title: "Computer Vision",
    projects: [
      projects.which_char,
      projects.find_char,
      projects.positional_cnn,
      projects.find_all_chars,
      projects.dense_cap,
      projects.mnist_gan,
      projects.colorizer,
      projects.find_all_chars_v2,
      projects.image_attention,
      projects.style_transfer,
      projects.style_gan,
    ],
  },
  nlp: {
    title: "Natural Language Processing",
    projects: [
      projects.next_char,
      projects.word2vec,
      projects.next_word,
      projects.word_attention,
      projects.word2vec_v2,
      projects.next_sentence,
    ],
  },
  reinforce: {
    title: "Reinforcement Learning",
    projects: [
      projects.tictactoe,
      projects.gridworld,
      projects.ping_ping,
      projects.racer,
      projects.dodger,
    ],
  },
  unsupervised: {
    title: "Unsupervised Learning",
    projects: [
      projects.auto_encoder,
      projects.self_organizing_map,
      projects.associative_memory,
    ],
  },
  misc: {
    title: "Miscellaneous",
    projects: [projects.spiking_neurons, projects.mnist_detection_dataset],
  },
}

export const projectCategories = [
  categoriesMap.linear,
  categoriesMap.vision,
  categoriesMap.nlp,
  categoriesMap.reinforce,
  categoriesMap.unsupervised,
  categoriesMap.misc,
]

export const orderedProjects = projectCategories
  .map(category => category.projects)
  .flat()

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

export const projectStatus = {
  toDo: "Yet to start",
  inProgress: "In Progress",
  done: "Done",
}

const projectsRaw = {
  learn_line: {
    title: "Learn a Line",
    desc: "Train a neuron to learn line equation",
    image: "learn_line.jpg",
    status: projectStatus.done,
    links: {
      app: "/gallery/linear/learn_line",
      source: "https://github.com/Akhilez/ml_gallery",
    },
  },
  linear_classifier: {
    title: "Linear Classifier",
    desc: "Train a neuron to classify data",
    image: "linear_classifier.png",
    status: projectStatus.done,
    links: {
      app: "/gallery/linear/linear_classifier",
      source: "https://github.com/Akhilez/ml_gallery",
    },
  },
  learn_curve: {
    title: "Learn A Curve",
    desc: "Train a neural net to predict curves",
    image: "learn_curve.png",
    status: projectStatus.done,
    links: {
      app: "/gallery/linear/learn_curve",
      source: "https://github.com/Akhilez/ml_gallery",
    },
  },
  deep_iris: {
    title: "Deep Iris",
    desc: "Train a neural net to classify Iris dataset",
    image: "deep_iris.png",
    status: projectStatus.done,
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
    status: projectStatus.done,
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
    status: projectStatus.toDo,
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
    status: projectStatus.done,
    links: {
      app: "https://akhilez.com/colorizer/",
      source: "https://github.com/Akhilez/ml_gallery",
    },
  },
  find_char: {
    title: "Find The Number",
    desc: "Draw a number in the box and find its location",
    image: "find_number.png",
    status: projectStatus.done,
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
    image: "positional_cnn.png",
    status: projectStatus.done,
    links: {
      app: "/gallery/vision/positional_cnn",
      source:
        "https://github.com/Akhilez/ml_gallery/blob/master/ml_py/app/vision/positional_mnist/positional_mnist.py",
      colab:
        "https://colab.research.google.com/github/Akhilez/ml_gallery/blob/master/ml_py/app/vision/positional_mnist/positional_mnist_colab.ipynb",
    },
  },
  dense_cap: {
    title: "Dense Cap",
    desc: "Generate English captions for numbers and their clusters",
    image: "under_construction2.png",
    status: projectStatus.toDo,
    links: {
      app: "/gallery/coming_soon",
    },
  },
  find_all_chars: {
    title: "Find All Numbers",
    desc: "",
    image: "faster_rcnn.png",
    status: projectStatus.toDo,
    links: {
      app: "/gallery/vision/find_all_chars",
      source:
        "https://colab.research.google.com/github/Akhilez/ml_gallery/blob/master/ml_py/app/vision/detector_v1/colab_detection.ipynb",
      colab:
        "https://colab.research.google.com/github/Akhilez/ml_gallery/blob/master/ml_py/app/vision/detector_v1/colab_detection.ipynb",
    },
  },
  image_attention: {
    title: "Attention, Attention!",
    desc: "",
    image: "under_construction2.png",
    status: projectStatus.toDo,
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  style_transfer: {
    title: "Style, Please",
    desc: "",
    image: "under_construction2.png",
    status: projectStatus.toDo,
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  style_gan: {
    title: "Style, Please V2",
    desc: "Style GAN",
    image: "under_construction2.png",
    status: projectStatus.toDo,
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
    status: projectStatus.done,
    links: {
      app: "/gallery/nlp/next_char",
      source:
        "https://github.com/Akhilez/ml_gallery/blob/master/ml_py/app/nlp/next_char/next_char.py",
      colab:
        "https://colab.research.google.com/github/Akhilez/ml_gallery/blob/master/ml_py/app/nlp/next_char/next_char_colab.ipynb",
    },
  },
  what_genre: {
    title: "What Genre - Attention",
    desc:
      "Lets predict what genre a movie is based on its plot using Attention!",
    image: "under_construction2.png",
    status: projectStatus.inProgress,
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  action_assistant: {
    title: "Action Assistant",
    desc: "Take actions by ordering in natural English language.",
    image: "under_construction2.png",
    status: projectStatus.toDo,
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  next_sentence: {
    title: "Next Sentence",
    desc: "",
    image: "under_construction2.png",
    status: projectStatus.toDo,
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },

  // Reinforcement Learning
  alpha_nine: {
    title: "Nine Mens Morris",
    desc: "9 Men's Morris with AlphaGo like algorithm",
    image: "alpha_nine.png",
    status: projectStatus.inProgress,
    links: {
      app: "/gallery/rl/alpha_nine",
      source: "",
    },
  },
  gridworld: {
    title: "Grid World",
    desc: "Navigate the player to a desired location.",
    image: "gridworld.png",
    status: projectStatus.toDo,
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  dodger: {
    title: "Dodger",
    desc: "Escape the on-coming obstacles by moving left or right",
    image: "under_construction2.png",
    status: projectStatus.toDo,
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
    status: projectStatus.toDo,
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  self_organizing_map: {
    title: "Self-Organizing Feature Maps",
    desc: "",
    image: "under_construction2.png",
    status: projectStatus.toDo,
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  associative_memory: {
    title: "Memorize Please",
    desc: "",
    image: "under_construction2.png",
    status: projectStatus.toDo,
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
    status: projectStatus.toDo,
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
  mnist_detection_dataset: {
    title: "MNIST Detection Dataset",
    desc: "",
    image: "under_construction2.png",
    status: projectStatus.inProgress,
    links: {
      app: "/gallery/coming_soon",
      source: "",
    },
  },
}
Object.keys(projectsRaw).map(key => {
  projectsRaw[key].id = key
})
export const projects = projectsRaw

const categoriesMap_ = {
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
    desc:
      "Visualize and learn Computer Vision models from basic classifiers to complex vision problems",
    projects: [
      projects.which_char,
      projects.find_char,
      projects.positional_cnn,
      projects.find_all_chars,
      projects.dense_cap,
      projects.mnist_gan,
      projects.colorizer,
      projects.image_attention,
      projects.style_transfer,
      projects.style_gan,
    ],
  },
  nlp: {
    title: "Natural Language Processing",
    desc: "Learn how neural networks are trained on text",
    projects: [
      projects.next_char,
      projects.what_genre,
      projects.action_assistant,
      projects.next_sentence,
    ],
  },
  reinforce: {
    title: "Reinforcement Learning",
    projects: [projects.gridworld, projects.alpha_nine, projects.dodger],
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
Object.keys(categoriesMap_).map(key => {
  categoriesMap_[key].category = key
})
export const categoriesMap = categoriesMap_

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

export const papers = {
  dis_cease: {
    title: "DisCease: Evolving Social Distancing And Herd Immunity",
    abstract:
      "I tried to create a model that simulates the spread of a disease that does not have a medicine. I observed the effects of varying parameters, then introduced a concept of deflections which mimic social distancing and social gatherings. I finally made an attempt to evolve these deflections based on a performance metric.",
    image: "dis_cease.png",
    link: "https://storage.googleapis.com/akhilez/papers/dis_cease.pdf",
    conference: "Complex Systems And Networks",
    association: "University of Cincinnati",
  },
  rl_survey: {
    title: "A Brief Survey of Model-Free Deep Reinforcement Learning",
    abstract:
      "Deep Reinforcement Learning is a branch of machine learning techniques that is used to find out the best possible path given a situation. It is an interesting domain of algorithms ranging from basic multi-arm bandit problems to playing complex games like Dota 2. This paper surveys the research work on model-free approaches to deep reinforcement learning like Deep Q Learning, Policy Gradients, Actor-Critic methods and other recent advancements.",
    image: "rl_survey.png",
    link: "https://storage.googleapis.com/akhilez/papers/rl_survey.pdf",
  },
}

export const papersList = [papers.dis_cease, papers.rl_survey]

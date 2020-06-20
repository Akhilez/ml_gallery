DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


tensorflowjs_converter --input_format keras \
                       $DIR/models/mnist_classifier.h5 \
                       $DIR/../../../../ml_js/src/vision/which_char/models

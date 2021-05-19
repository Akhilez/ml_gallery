_Train a neuron to learn line equation_


---


To learn neural networks (aka deep learning),
one must start with the smallest unit - a neuron.
In this project, you’ll learn the basics of a neuron by training it on a straight line equation: `y = mx + c`


**************
**************


> ### Preface:
> It’s good to have a machine learning background to get started with deep learning, but it is not necessary!
> All you need to know to get started at the moment is that you send in some input numbers and out comes some numbers.
> The training part is to iteratively make sure that we get the right outputs.\
> Alright, so when it comes to learning a straight line, your inputs would be `x` values `[x1, x2, x3, ...]`
> and the outputs SHOULD be `[y1, y2, y3, ...]`.
> I used “should be” because the neural networks may predict erroneous outputs `[y1 + e1, y2 + e2, y3 + e3, ...]`.
> So the goal of training is to minimize the error, that is, `e1 = 0, e2 = 0, e3 = 0 …`.
> Now, all you need to know is `m` and `c` to predict the correct `y` values.
> In deep learning terms, `m` and `c` are called **weights** or **parameters**.


Here’s the task we are trying to solve:
Given a bunch of `(x, y)` pairs on a line `[(x1, y1), (x2, y2), ...]`,
we must find the parameters `m` and `c`.

You must’ve learnt in your school how to find the slope `m` of a straight line using 2 points
`(x1, y1)` and `(x2, y2)` using the formula `m = (x2 - x1) / (y2 - y1)`.
And how would you find the **y intercept**? - by substituting `x = 0` in the line equation `y = mx + c`.\
Cool, instead of using these formulas, let's use a _neuron_!


mylist <- c(1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1)
data <- matrix(mylist, nrow=4, ncol=3)
print(data)
target <- matrix(c(1, 0, 1, 0), nrow=4)
print(target)

sigmoid <- function(x, deriv) {
    if (deriv == TRUE) {
        return (x*(1-x))
        }
    return (1/(1+exp(-x)))
    }
w0 = matrix(c(0.1, -0.1, 0.01), nrow=3)
learning_rate <- 1
for (i in 1:10000) {
    layer0 <- data
    layer1 <- sigmoid(layer0%*%w0, FALSE)
    error <- layer1-target
    delta <- error*sigmoid(layer1, TRUE)
    w0 <- w0 - learning_rate*t(layer0)%*%delta
}
print(layer1)
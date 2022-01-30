require(farff)
require(RWeka)
set.seed(42)

dataset <- readARFF("data/esl.arff")

# change the label variable name to class
names(dataset)[dim(dataset)[2]] <- "class"

# Convert to ordered factor the class variable
dataset$class <- factor(dataset$class, ordered = T)

test_rows <- sample(1:nrow(dataset), 100)
train_dataset <- dataset[-test_rows,]
test_dataset <- dataset[test_rows,]

number_classes <- length(levels(dataset$class))

dataset_list <- lapply(
    levels(dataset$class)[-number_classes],
    function (i) {
        new_dataset <- train_dataset
        new_dataset$newClass <- factor(ifelse(new_dataset$class > i, 1, 0))
        new_dataset$class <- NULL

        return(new_dataset)
    }
)

naive_bayes <- make_Weka_classifier("weka/classifiers/bayes/NaiveBayes")

model_list <- lapply(
    dataset_list,
    function (dataset, classifier = naive_bayes) {
        classifier(newClass~., data=dataset)
    }
)

predictions <- lapply(
    model_list,
    function (generic_model) {
        predict(generic_model, newdata = test_dataset, type = "probability")
    }
)

meow <- apply(
    array(unlist(predictions), dim = c(dim(test_dataset)[1], 2, length(model_list))),
    c(3, 1),
    function (a) {
        a
    }
)


# need to work on this
apply(
    meow,
    3,
    function (x) {
        transposed <- t(x)
    }
)

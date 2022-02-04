require(farff)
require(readr)
require(RWeka)
set.seed(42)

get_train_test_dataset <- function(dataset_name) {
    dataset <- readARFF(dataset_name)

    # change the label variable name to class
    names(dataset)[dim(dataset)[2]] <- "class"

    # Convert to ordered factor the class variable
    dataset$class <- factor(dataset$class, ordered = T)

    test_rows <- sample(1:nrow(dataset), 100)

    return(
        list(
            train = dataset[-test_rows,],
            test = dataset[test_rows,]
        )
    )
}


train_ordinal <- function (classifier, train_dataset) {
    number_classes <- length(levels(train_dataset$class))

    # In this step, number_classes - 1 datasets are generated
    dataset_list <- lapply(
        levels(train_dataset$class)[-number_classes],
        function (i) {
            new_dataset <- train_dataset
            new_dataset$newClass <- factor(ifelse(new_dataset$class > i, 1, 0))
            new_dataset$class <- NULL

            return(new_dataset)
        }
    )

    # The list of datasets is iterated and a list of models is returned
    model_list <- lapply(
        dataset_list,
        function (dataset) {
            classifier(newClass~., data=dataset)
        }
    )

    return(model_list)
}

predict_ordinal <- function(models, test) {
    # the probability for each instance from each model is calculated
    prediction_probs <- lapply(
        models,
        function (generic_model) {
            predict(generic_model, newdata = test, type = "probability")
        }
    )

    # Get a 3-d matrix in which the third dimension represents each test instance
    # the second dimension represents the probabilities for each class
    # and the first dimension represents the probabilities for each model
    predictions_instance_oriented <- apply(
        array(unlist(prediction_probs), dim = c(dim(test)[1], 2, length(models))),
        c(3, 1),
        function (a) {
            a
        }
    )

    # Iteration per instance (3rd dimension)
    # We apply the transpose function to the output of the apply function
    # to represent the test instances as the rows and the probabilities of each
    # class as the columns
    class_probs <- t(apply(
        predictions_instance_oriented,
        3,
        function (x) {
            # we transpose the matrix to have the probabilities of each model per row
            transposed <- t(x)

            final_probabilities <- NULL

            # first we get the probability of belonging to the first class
            final_probabilities <- c(
                final_probabilities,
                transposed[1, 1]
            )

            # Then we get the probabilities for the classes in the range [2, number_classes - 1]
            # number_classes - 1 == length(models)
            for (i in 2:length(models)) {
                final_probabilities <- c(
                    final_probabilities,
                    transposed[i - 1, 2] * transposed[i, 1]
                )
            }

            # Finally, we get the probability for the last class
            final_probabilities <- c(
                final_probabilities,
                transposed[dim(transposed)[1], 2]
            )

            return(final_probabilities)
        }
    ))

    # The level name is taken for each test instance
    prediction_label <- apply(
        class_probs,
        1,
        function (row) {
            levels(test$class)[which.max(row)]
        }
    )

    return(prediction_label)
}

naive_bayes <- make_Weka_classifier("weka/classifiers/bayes/NaiveBayes")
datasets <- get_train_test_dataset("data/swd.arff")

model_list <- train_ordinal(naive_bayes, datasets$train)
predictions <- predict_ordinal(model_list, datasets$test)
predictions

# MZE
1 - mean(predictions == datasets$test$class)

# Mean zero one error has been used to measure the quality of the ordinal classifier
# In this metric, the smaller the value is, the better the classifier is.
# It is calculated as 1 - accuracy

# era.arff dataset
# The MZE obtained for this dataset is 0.7

# esl.arff dataset
# The MZE obtained for this dataset is 0.45

# lev.arff dataset
# The MZE obtained for this dataset is 0.39

# swd.arff dataset
# The MZE obtained for this dataset is 0.46

require(xgboost)
require(farff)
require(readr)
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

train_monotonic <- function (train_dataset) {
    number_classes <- length(levels(train_dataset$class))

    # We generate number-classes - 1 datasets
    # In this case, to generate the label, those who have a 1
    # are the classes that are equal or greater than the class
    # that is taken in account in each iteration
    dataset_list <- lapply(
        levels(train_dataset$class)[2:number_classes],
        function (i) {
            new_dataset <- train_dataset
            new_dataset$newClass <- factor(ifelse(new_dataset$class >= i, 1, 0))
            new_dataset$class <- NULL

            return(new_dataset)
        }
    )

    # The data is transform to a DMatrix in order to be handled by xgboost
    model_list <- lapply(
        dataset_list,
        function (dataset) {
            dmatrix <- xgb.DMatrix(
                data=as.matrix(dataset[, -dim(dataset)[2]]),
                label=as.vector(dataset$newClass)
            )

            # Then the monotone constraints are taken into account
            # as well as the objective, a logistic regression
            # a 10 rounds iteration is set
            xgb.train(
                params = list(
                    monotone_constraints = 1,
                    objective = "binary:logistic"
                ),
                data=dmatrix,
                nrounds=10
            )
        }
    )

    return(model_list)
}

predict_monotonic <- function(models, test) {
    # Then the predictions for each test instance from each model are calculated
    prediction_scores <- sapply(
        models,
        function (generic_model) {
            dmatrix_test <- xgb.DMatrix(
                data = as.matrix(test[, -dim(test)[2]])
            )

            predict(generic_model, newdata = dmatrix_test)
        }
    )

    # After that, the predictions labels are transformed to 0s and 1s
    prediction_labels <- ifelse(prediction_scores > 0.5, 1, 0)

    # And finally,the 1s are added and we take in account the first class by adding 1
    # Then with that sum, we get the level name
    final_prediction_labels <- apply(
        prediction_labels,
        1,
        function (x) {
            levels(test$class)[sum(x) + 1]
        }
    )

    return(final_prediction_labels)
}

datasets <- get_train_test_dataset("data/swd.arff")

model_list <- train_monotonic(datasets$train)
predictions <- predict_monotonic(model_list, datasets$test)
predictions

# MZE
1 - mean(predictions == datasets$test$class)

# Mean zero one error has been used to measure the quality of the ordinal classifier
# In this metric, the smaller the value is, the better the classifier is.
# It is calculated as 1 - accuracy

# era.arff dataset
# The MZE obtained for this dataset is 0.74

# esl.arff dataset
# The MZE obtained for this dataset is 0.24

# lev.arff dataset
# The MZE obtained for this dataset is 0.38

# swd.arff dataset
# The MZE obtained for this dataset is 0.36

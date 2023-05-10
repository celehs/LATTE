pak.list = c('data.table','MASS','pROC','dplyr','ff','MAP')

for (pak in pak.list){

    yo = require(pak,character.only = T)
    if (!yo)
    {
        install.packages(pak,repos = "http://cran.us.r-project.org")
        require(pak,character.only = T)
    }
}
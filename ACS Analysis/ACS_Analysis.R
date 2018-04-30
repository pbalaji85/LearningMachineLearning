library(tidyverse); library(ggfortify); library(factoextra)
## Read in the first 100 rows to determins the classes of the columns
a <- read.delim('/Users/balaji/Downloads/csv_pus/ss16pusa.csv', header = T, 
                sep=",", nrows=100)

## I read through the dictionary for the 2016 ACS and decided to remove all the 
## flag and weight data (for now, we will ignore whether the values are 
## which values are weighted or imputed)

## Get the class of each column in the dataset
classes <- apply(a,2,class)
names(classes) <- NULL

## Based on the dictionary, 'FAGEP' is the column where the flag and weights
## data begin
idx <- grep('FAGEP',colnames(a))
## Setting the classes of the columns starting at 'FAGEP' to NULL, so they 
## read into memory
classes[idx:length(classes)] <- 'NULL'

## Read in the data files with the flag and weights information omitted
a <- read.delim('/Users/balaji/Downloads/csv_pus/ss16pusa.csv', header = T, 
                na.strings="", sep = ",", colClasses = classes)
b <- read.delim('/Users/balaji/Downloads/csv_pus/ss16pusb.csv', header = T,
                na.strings="", sep = ",", colClasses = classes)

## Merge the two files
acs <- rbind(a,b)
## clean up to remove the individual files
rm(a,b,classes,idx)

## Since the focus of this study is adults, let's remove all records involving
## individuals less than 18 years old
acs_over18 <- acs[acs$AGEP>18,]  

## Now Let's calculate the percentage of missing values in each feature. We could
## eliminate features that are largely composed of empty values
rows <- nrow(acs_over18)
na_prop = apply(acs_over18,2,function(x){((sum(is.na(x)))/rows)*100})

## Remove all features that have over 40% missing values
over40_NA <- names(na_prop)[(na_prop >= 40)]
acs_over18 <- acs_over18[,!names(acs_over18) %in% over40_NA]

## Remove categorical features that deal with various types of disability...
## including Eye, Ear, mobility etc..
disable <- grep('^D',colnames(acs_over18))
acs_over18 <- acs_over18[,-disable]

## Remove duplicated features that deal with marital status. I am retaining MAR
## which contains information about the martical status of the individual
marital <- grep('^MAR.',colnames(acs_over18))
acs_over18 <- acs_over18[,-marital]

## Remove features that deal with detailed classification of race. I am retaining
## one feature for race that preserves top-level race information.
race <- grep('^RAC', colnames(acs_over18))
acs_over18 <- acs_over18[,-race[2:length(race)]]

## Other features that were chosen to be discarded based on a reading of the
## dictionary. These features are either categorical, or indicate sub-classification
## of a feature already present in the dataset
erase_features <- c('RC','RELP','RT','NATIVITY','GCL','INTP','MIG',
                 'NWAB','NWRE','OIP','SCH','SSIP','WKW','WRK','ANC',
                 'ANC1P','ANC2P','ESR','FHICOVP','HICOV','HISP','MSP','NAICSP',
                 'POBP','QTRBIR','SOCP','HINS5','HINS6','HINS7','OCCP','OC',
                 'ADJINC','SPORDER', 'WAOB', 'PRIVCOV','PUBCOV','INDP')

acs_over18 <- acs_over18[,!names(acs_over18) %in% erase_features]

## We finally have all of the features we need for our analysis.
## An advantage of the ACS dataset is that most features have already been
## engineered to numeric. Let's conver them all to numeric
acs_over18 <- as.data.frame(apply(acs_over18, 2, as.numeric))



## Let's once again take a look at the proportion of NA values in each of the
## features
rows <- nrow(acs_over18)
na_prop = apply(acs_over18,2,function(x){((sum(is.na(x)))/rows)*100})
print(na_prop)

## Since most features have been encoded, there are only 3 features that contain
## NAs. 
## COW feature is class of worker, NAs is people who worked more than 5 years go
## or never worked. Let's encode the NAs in COW to 0
acs_over18$COW[is.na(acs_over18$COW)] <- 0

## WKHP is the number of hours worked per week, the NAs represent people who did
## not work. Let's set the NA value for this feature to 0
acs_over18$WKHP[is.na(acs_over18$WKHP)] <- 0

## POVPIP is an encoded value poverty to income ratio from 0 to 501. Let's
## encode the NAs for this feature as -1
acs_over18$POVPIP[is.na(acs_over18$POVPIP)] <- -1

plot( acs_over18$AGEP, as.factor(acs_over18$ST))
## I'm going to try and normalize the variables dealing with income by applying 
## the adjustment factor
income_features <- c('SSP','WAGP','PINCP','PERNP','SEMP','RETP')

acs_over18[,names(acs_over18) %in% income_features] <- 
        acs_over18[,names(acs_over18) %in% income_features]/1007588



## Perform K-mean clustering of the entire dataset. Start with an arbitrary K
## assignment of 10 clusters
km1 <- kmeans(acs_over18[,4:ncol(acs_over18)], 10)
png(filename = '/Users/balaji/Downloads/csv_pus/unsum_kmeans.png', width = 838, height = 525)
autoplot(km1, data = acs_over18[,4:ncol(acs_over18)], label = T, label.size = 3)
dev.off()

## Summarize data over PUMA codes 
acs_over18 <- as.tibble(acs_over18) 

puma_summary <- acs_over18[,c(2,4:ncol(acs_over18))] %>% group_by(PUMA) %>%
        summarize_all(funs(median))

# Removing PUMA feature from the data frame (data summarized by PUMA region)
tmp = as.data.frame(puma_summary)
rownames(tmp) = tmp$PUMA
tmp = tmp[,-1]


# Performing K-mean clustering using the summarized data
km = kmeans(tmp,10)
autoplot(km, data = tmp, label = TRUE, label.size = 3)

#elbow method

set.seed(123)

fviz_nbclust(tmp, kmeans, method = "wss")

set.seed(123)
final_km <- kmeans(tmp, 5)
print(final_km)
autoplot(final_km, data = tmp, label = F, label.size = 3)


finalKM_summaryStats = tmp %>%
                         mutate(Cluster = final_km$cluster) %>%
                        group_by(Cluster) %>%
                        summarise_all("median")

###PCA
# Removing features that show 0 variance
tmp1 = tmp[,apply(tmp,2,var)!=0]
pca <- prcomp(tmp1,scale. = T)

png(filename = '/Users/balaji/Downloads/csv_pus/PCA_by_puma.png', width = 838, height = 525)
infodf <- data.frame(featurename = rownames(tmp1),st=as.factor(acs_over18$ST[match(rownames(tmp1),acs_over18$PUMA)]))
autoplot(pca, data=infodf,label.label='featurename',label.size=4,label=T,colour="st")
dev.off()

## histogram of WAGP (wages/salary income past 12 months) per state
uStates = unique(acs_over18$ST)
maxWAGP = max(acs_over18$WAGP)
for(i in 1:length(uStates))
{
  temp = acs_over18[acs_over18$ST==uStates[i],]
  hist(temp$WAGP,xlim=c(0,maxWAGP),las=1,breaks=100)
  boxplot(temp$WAGP)
}

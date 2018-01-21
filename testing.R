

# import libraries
library(lme4)

# import data
df = read.csv('training_data.csv')
print(df)


# Fit model --
model = glmer(cbind(num_dropped, enrolled-num_dropped) ~ (1|nct_id) + ., family=binomial, data=df)


fixef(model)
vcov(model)
cov2cor(vcov(model))

######

library(rstanarm)

df = read.csv('training_data.csv')

model = stan_glmer(cbind(num_dropped, enrolled-num_dropped) ~ (1|nct_id) + duration + is_cancer, family=binomial, 
                   data=df,
                   chains=1,
                   refresh=1)


model = stan_glmer(cbind(num_dropped, enrolled-num_dropped) ~ (1|nct_id) + duration + is_cancer, 
                   family=binomial, 
                   data=df,
                   algorithm='meanfield')

# 'QR' argument?
# plogis
# posterior_interval(model, prob=0.95)
# mean(posterior_linpred(model, transform=TRUE, newdata=df[100,]))
# mean(posterior_linpred(model, transform=TRUE, newdata=df[100,], re.form=NA))
# 

write.csv(fixef(model), 'test_coefs.csv')
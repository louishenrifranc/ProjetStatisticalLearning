# subset(data, prof = "sans emploi")
# generer nombre aléatoire: rbinom, rgamma...

# QUESTION A
x = c(68,70,65,55,70,72,75)
y = c(168,172,165,160,180,182,175)
plot(x,y)
x = scale(x)

# QUESTION B
len = length(x)
y_sum = sum(y)
x_sum = sum(x)
x_sum_squared = sum(x*x)
x_mean = x_sum/len
y_mean = y_sum/len


# Modèle 1
beta_o = y_sum/len


# Modèle 2
beta_o2 = y_sum/x_sum

# Modele 3
# (sum((x-x_mean)*(y-y_mean)))
a = (sum((x - x_mean) * (y - y_mean)))/(sum((x - x_mean)^2))
b = (y_mean - a*x_mean)

# Add a line that pass through the data
abline(b,a)
# Check model correctness
matrix_final = matrix(c(y,rep(y_mean,len),beta_o2*x,b+a*x),nrow = len,byrow = FALSE)

# QUESTION C
bo = 150:190
y1 = (sum(y) - len*bo)^2
plot(bo,y1)
# On remarque que le minimum est celui que l'on a trouvé 

# QUESTION D
x = c(68,70,65,55,70,72,75)
y = c(168,172,165,160,180,182,175)

u = 100:200
v = seq(0,2,0.01)
#f = outer(u,v,(function(u,v)(sum((y - u - v*x)^2))))

#sapply(u,v,f(u,v))

#contour(u,v,f, nlevels = 40)
#persp(u,v,f)

# QUESTION E
dat = data.frame(x = c(68,70,65,55,70,72,75),
                 y = c(168,172,165,160,180,182,175))
mse <- function(betao,beta1,Y=dat$y,x=dat$x) {
  mean((Y - betao - beta1*x))
}
sapply(seq(from=0.10,to=0.15,by=0.01),mse,beta1=1)
mse.plottable <- function(betao,...){sapply(a,mse,...)}

curve(mse.plottable(betao=x,beta1=6611),from=0.10,to=0.15)






result <- optim(par = c(0, 1), min.RSS, data = dat)
plot(y ~ x, data = dat, main="Least square regression")
abline(a = result$par[1], b = result$par[2], col = "red")



#intervalles de confiane beta0,beta a 96  %
# confint(# mettre l'objet de lm)
  
# normal Q_Q : verfie que les epsilon sont de loi normale N(0,sigma)


mu1 = 0
sigma1 = 1

mu2 = 1
sigma2 = 1

x = (-6:0.01:6)
y1 = (1/(sigma1*sqrt(2*pi)))*exp(-0.5*((x-mu1)/sigma1).^2)
y2 = (1/(sigma2*sqrt(2*pi)))*exp(-0.5*((x-mu2)/sigma2).^2)
y = (y1-y2==0);
plot(x,y1,'color','r'); hold on
plot(x,y2,'color','b'); hold on
plot(x,y)
mse_kidney <- readMSIData('data/kidney.imzml')

mse_peakref <- mse_kidney %>%
  peakAlign() %>%
  peakFilter() %>%
  process()

# bin profile spectra to peaks
mse_peaks <- mse_kidney %>%
  normalize()%>%
  peakBin(mz(mse_peakref))%>%
  process()


spec = spectra(mse_peaks)
coords = coord(mse_peaks)
coords = as.data.frame(as.matrix(t(coords)))
spec = as.data.frame(spec)
spec = spec[!duplicated(as.list(coords))]
coords = coords[!duplicated(as.list(coords))]
coords = as.data.frame(as.matrix(t(coords)))

label <- rep(1, dim(mse_peaks)[1])
write.table(spec, file="kidney/spec.csv", row.names = F, col.names = F)
write.table(label, file="kidney/label.csv", row.names = F, col.names = F)



mse_peaks <- MSImageSet(spectra = as.matrix(spec), coord = coords, mz=mz(mse_peaks))



mse_peaks<-as(mse_peaks, 'MSImagingExperiment')

load('data/kidney.rdata')

dc_pre2 <- read.csv('kidney/dc_pre.txt', header=F,sep=' ')
cluster<-as.character(unlist(dc_pre2))

mse_clust = mse_peaks[cluster=='3',]
ssc <- spatialShrunkenCentroids(mse_clust, k=7, r=2, s=0)
image(ssc)

colors = c('#A06AB4','#FFD743','#07BB9C', '#D773A2', '#FABEA7', '#9CADA4', '#FBE2E5', 'gray', 'blue', 'purple')
mcol = colors[as.numeric(cluster)+1]

for (i in 1:length(unique(cluster)))
{
  print(plot(mse_peaks[cluster==(i-1),], pixel=100, col = mcol[cluster==(i-1)],xlim=c(100,2000)))
}

mcols = c()

for (i in 1:dim(mse_peaks)[1])
{
  mcols[i] = colors[as.numeric(cluster[i])+1]
}

plot(mse_peaks, pixel=100, col = mcols)

data = data.frame(mz = mz(mse_peaks), intensity = rowMeans(spectra(mse_peaks)), cluster = as.factor(cluster))

ggplot(data, aes(x=mz, y=intensity))+
  geom_point(aes(color=cluster, shape = cluster))+
  scale_shape_manual(values=1:10)+
  labs(x= 'm/z', y='Intensity')+
  theme_bw()

dc_prob <- read.csv('kidney/dc_prob.txt', header=F,sep=' ')
cluster_ = c()
for (i in 1:dim(dc_prob)[1])
{
  x = which(dc_prob[i,] == max(dc_prob[i,]))
  cluster_[i] = x
}

for (i in 1:length(unique(cluster)))
   {
      print(plot(data$mz[cluster==(i-1)], data$intensity[cluster==(i-1)], type = 'l', col = mcol[cluster==(i-1)],
                 xlab = 'm/z', ylab = 'Intensity'))
}


for (i in 1:length(unique(cluster)))
{
  mean_spec<-colMeans(spectra(mse_peaks)[which(cluster==as.character(i-1)),])
  print(image(mse_peaks,mean_spec~x*y))
}


dc_pre2 <- read.csv('kidney/dc_pre_patch.txt', header=F,sep=' ')
cluster<-as.character(unlist(dc_pre2))

non_id <- read.csv('kidney/non_id.txt', header=F,sep=' ')
non_id = non_id+1
non_id<-as.character(unlist(non_id))

cluster_ = rep(10, dim(mse_peaks)[1])
cluster_[!((1:dim(mse_peaks)[1])%in%non_id)]=cluster
cluster=cluster_


###################SSC plot

cluster = as.numeric(cluster)+1

ssc_total <- spatialShrunkenCentroids(mse_peaks, k = 6, r = 1, s=0)
image(ssc_total)

ssc_propose_kidney = list()

for (i in 1:length(unique(cluster)))
{
  mse_sub <- mse_peaks[cluster==i,]
  ssc <- spatialShrunkenCentroids(mse_sub, k = 6, r = 1, s=0)
  print(image(ssc))
  ssc_propose_kidney[[i]] <- ssc
}

#########################plot spatial segmentation and mean ion images for each cluster
myColor = c(rgb(204/255,102/255,92/255), rgb(193/255,218/255,106/255), rgb(121/255,208/255,217/255),rgb(96/255,131/255,212/255), rgb(187/255,94/255,213/255),'gray')

par(mfrow=c(1,2))
for (i in 1:length(unique(cluster)))
{
  mean_spec <- colMeans(spectra(mse_peaks)[cluster==i,])
  print(image(mse_peaks, mean_spec~x*y,axes = FALSE, colorkey=FALSE,xlab=NA,ylab=NA,layout=FALSE))
  print(image(ssc_propose_kidney[[i]], col = myColor,axes = FALSE, key=FALSE,xlab=NA,ylab=NA, layout=FALSE))
  
}

colors = c('#A06AB4','#FFD743','#07BB9C', '#D773A2', '#FABEA7', '#9CADA4', '#FBE2E5', 'gray', 'blue', 'purple')
mcol = colors[cluster]

intensY_ <- rowMeans(spectra(mse_peaks))
mzX_ <- mz(mse_peaks)

par(mfrow=c(2,2))
for (i in 1:length(unique(cluster)))
{
  intensY=c()
  mzX=c()
  Ytmp = intensY_[cluster==i]
  Xtmp = mzX_[cluster==i]
  for (j in 1:length(Ytmp))
  {
    intensY <- c(intensY, c(0,Ytmp[j],0))
  }
  
  for (j in 1:length(Xtmp))
  {
    mzX <- c(mzX, rep(Xtmp[j],3))
  }
  
  print(plot(mzX,intensY, col = mcol[cluster==i],xlim=range(mzX_),type='l', xlab='m/z',ylab='Intensity'))
  
}

############################ new plot
myColor = c(rgb(204/255,102/255,92/255), rgb(193/255,218/255,106/255), rgb(121/255,208/255,217/255),rgb(96/255,131/255,212/255), rgb(187/255,94/255,213/255),'gray')

# Segment 2
myColor = c('white', rgb(193/255,218/255,106/255), rep('white', 4))

print(image(ssc_propose_kidney[[2]], value = 'class',col = myColor,axes = FALSE, key=FALSE,xlab=NA,ylab=NA, layout=FALSE))

myColor = c(rep('white', 5), 'grey')

print(image(ssc_propose_kidney[[2]], value = 'class',col = myColor,axes = FALSE, key=FALSE,xlab=NA,ylab=NA, layout=FALSE))


######################## Cluster 10

par(mfrow=c(1,2))

print(image(ssc_propose_kidney[[10]], value = 'class',col = myColor,axes = FALSE, key=FALSE,xlab=NA,ylab=NA, layout=FALSE))


# Segment 2
myColor = c('white', rgb(193/255,218/255,106/255), rep('white', 4))

print(image(ssc_propose_kidney[[10]], value = 'class',col = myColor,axes = FALSE, key=FALSE,xlab=NA,ylab=NA, layout=FALSE))



# segment 4

myColor = c(rep('white', 3),rgb(96/255,131/255,212/255), rep('white',2))
print(image(ssc_propose_kidney[[10]], value = 'class',col = myColor,axes = FALSE, key=FALSE,xlab=NA,ylab=NA, layout=FALSE))

# segment 6

myColor = c(rep('white', 5),'gray')
print(image(ssc_propose_kidney[[10]], value = 'class',col = myColor,axes = FALSE, key=FALSE,xlab=NA,ylab=NA, layout=FALSE))


##################### cluster 7

######################## Cluster 10

par(mfrow=c(1,2))

myColor = c(rgb(187/255,94/255,213/255), rgb(193/255,218/255,106/255), rgb(121/255,208/255,217/255),rgb(96/255,131/255,212/255), rgb(204/255,102/255,92/255),'gray')
print(image(ssc_propose_kidney[[7]], value = 'class',col = myColor,axes = FALSE, key=FALSE,xlab=NA,ylab=NA, layout=FALSE))


# Segment 1
myColor = c(rgb(187/255,94/255,213/255),  rep('white', 5))

print(image(ssc_propose_kidney[[7]], value = 'class',col = myColor,axes = FALSE, key=FALSE,xlab=NA,ylab=NA, layout=FALSE))


# Segment 2
myColor = c('white', rgb(193/255,218/255,106/255), rep('white', 4))

print(image(ssc_propose_kidney[[7]], value = 'class',col = myColor,axes = FALSE, key=FALSE,xlab=NA,ylab=NA, layout=FALSE))


# Segment 6
myColor = c( rep('white', 5), 'gray')

print(image(ssc_propose_kidney[[7]], value = 'class',col = myColor,axes = FALSE, key=FALSE,xlab=NA,ylab=NA, layout=FALSE))


##################All Data

myColor = c(rgb(121/255,208/255,217/255),rgb(187/255,94/255,213/255),  rgb(204/255,102/255,92/255), rgb(204/255,102/255,92/255), rgb(96/255,131/255,212/255), rgb(193/255,218/255,106/255),'gray')
print(image(ssc, value = 'class',col = myColor,axes = FALSE, key=FALSE,xlab=NA,ylab=NA, layout=FALSE))

#### segment 1


myColor = c(rgb(121/255,208/255,217/255),rep('white', 6))
print(image(ssc, value = 'class',col = myColor,axes = FALSE, key=FALSE,xlab=NA,ylab=NA, layout=FALSE))

### segment 5

myColor = c(rep('white',4), rgb(96/255,131/255,212/255), rep('white',2))
print(image(ssc, value = 'class',col = myColor,axes = FALSE, key=FALSE,xlab=NA,ylab=NA, layout=FALSE))


### segment 6

myColor = c(rep('white',5), rgb(193/255,218/255,106/255), rep('white',1))
print(image(ssc, value = 'class',col = myColor,axes = FALSE, key=FALSE,xlab=NA,ylab=NA, layout=FALSE))


myColor = c(rep('white',6), 'grey')
print(image(ssc, value = 'class',col = myColor,axes = FALSE, key=FALSE,xlab=NA,ylab=NA, layout=FALSE))



df=read.csv("concept_regparam_decoding.csv",row.names=1)
rownames(df) = df[,"X0.concept_name"]
df = df[,-which(colnames(df) == "X0.concept_name")]

library(pheatmap)
pdf("concept_regparam_decoding.pdf",width=300,height=20)
pheatmap(df)
dev.off()

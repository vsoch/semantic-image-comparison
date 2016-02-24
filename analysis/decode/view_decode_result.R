library(pheatmap)
data = read.csv("data/concept_regparam_decoding.csv",row.names=1)

pdf("data/decoding_result.pdf",width=30,height=20)
pheatmap(data)
dev.off()

library(pheatmap)
data = read.csv("data/concept_regparam_decoding_named.csv",row.names=1)

pdf("data/decoding_result.pdf",width=60,height=20)
pheatmap(data)
dev.off()

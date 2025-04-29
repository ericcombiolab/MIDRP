library(TwoSampleMR)
library(devtools)
library(phenoscanner)
require(meta)
require(metafor)
source("./external_GWAS_process.R")


# dis = "CAD"
# disease_gwas_ids <- c("ebi-a-GCST005195", "ebi-a-GCST003116")
# is_file <- c(F, F)

# dis = "Type2-Diabetes"
# disease_gwas_ids <- c("ebi-a-GCST006867", "ukb-b-13806")
# is_file <- c(F, F)

dis = "BC"
disease_gwas_ids <- c("ieu-a-1126","ebi-a-GCST004988")
is_file <- c(F, F)

# read sumstats file
gwas_files <- list()
f <- 1
while (f <= length(disease_gwas_ids)){
  if (is_file[f]) {
    gwas_files[[sprintf("gwas%d", f)]] = read.csv(disease_gwas_ids[f], 
                                                  header = TRUE, 
                                                  sep = '\t')
  } else {
    gwas_files[[sprintf("gwas%d", f)]] = 'None'
  }
  f <- f+1
}

ao <- available_outcomes()
MRC_IEU_db = ao[grepl("MRC-IEU", ao$consortium),]
Neale_lab_db = ao[grepl("Neale", ao$author),]


for (phe in c('Lifestyle', 'Physical_measures')) {
  
  feature_file = sprintf("./%s/%s_selection_info.csv", 
                         dis, phe)
  feature_selection <- read.csv(feature_file, header = FALSE)
  
  
  ivw_data = data.frame()
  k <- 1
  while (k <= nrow(feature_selection)) {
    
    # search feature dataset id
    extract_id <- MRC_IEU_db[grepl(feature_selection[k,]$V3, MRC_IEU_db$trait),]
    if (nrow(extract_id) == 0) {
      extract_id <- Neale_lab_db[grepl(feature_selection[k,]$V3, Neale_lab_db$trait),]
      if (nrow(extract_id) == 0) {
        k <- k + 1
        next
      }
    }
    
    j <- 1
    while (j <= nrow(extract_id)) {
      
      id = extract_id[j,]$id
      fea_name = extract_id[j,]$trait
      
      snps_exposure_file = sprintf("./feature_selection/SNPs/%s.csv", id)
      if (file.exists(snps_exposure_file)){
        snps_exposure = read.csv(snps_exposure_file)
      }else{
        
        print(fea_name)
        # j <- j+1
        # next
        
        snps_exposure = extract_instruments(outcomes = id, p1 = 1e-05, p2 = 1e-05)
        
        # MAF > 0.01
        snps_exposure <- clump_data(snps_exposure)
        snps_exposure$exposure = paste(feature_selection[k,]$V3, snps_exposure$exposure)
        
        # phenoscanner
        del_row = c()
        p <- 1
        repeat{
          n = p * 100
          m = n - 100 + 1
          if (n > nrow(snps_exposure))
            n <- nrow(snps_exposure)
          if (n < m)
            break
          res <- phenoscanner(snpquery=snps_exposure[m:n,]$SNP, pvalue = 5e-8)$results
          q <- m
          while (q <= n) {
            nr = nrow(res[res$snp %in% snps_exposure[q,]$SNP,])
            if (nr > 1)
              del_row <- append(del_row, -q)
            if (nr == 1)
              print(snps_exposure[q,]$SNP)
            q <- q + 1
          }
          p <- p + 1
        }
        if (length(del_row) != 0)
          snps_exposure <- snps_exposure[del_row,]
        
        if (nrow(snps_exposure) == 0) {
          j <- j + 1
          next
        }
        
        write.csv(snps_exposure, file = snps_exposure_file, row.names = FALSE)
        #====================
      }
      
      
      tmp_res <- c()
      study_est <- c()
      study_se <- c()
      study_p <- c()
      study_lower <- c()
      study_uppper <- c()
      # MR
      f = 1
      while (f <= length(disease_gwas_ids)){
        if (is_file[f]){
          snps_disease <- external_GWAS_process_1(gwas_files[[sprintf("gwas%d", f)]], 
                                                  snps_exposure$SNP, 
                                                  beta_col = 'beta', 
                                                  eaf_col = 'maf', 
                                                  outcome=dis)
        }else {
          disease_gwas_id = disease_gwas_ids[f]
          snps_disease <- extract_outcome_data(snps = snps_exposure$SNP,
                                               outcomes = disease_gwas_id)
        }
        snps_disease = snps_disease[!duplicated(snps_disease$SNP),]
        if (nrow(snps_disease) == 0) {
          next
        }
        
        dat <- harmonise_data(exposure_dat = snps_exposure,
                              outcome_dat = snps_disease)
        dat <- dat[dat$mr_keep==TRUE,]
        dat2 <- dat_to_MRInput(dat)
        res <- MendelianRandomization::mr_ivw(dat2[[1]])
        study_est <- c(study_est, res$Estimate)
        study_se <- c(study_se, res$StdError)
        study_p <- c(study_p, res$Pvalue)
        study_lower <- c(study_lower, res$CILower)
        study_uppper <- c(study_uppper, res$CIUpper)
        res <- c(res$Estimate, res$CILower, res$CIUpper, res$StdError, res$Pvalue)
        tmp_res <- c(tmp_res, res)
        
        f <- f + 1
      }
      
      # Meta analysis
      gen <- metagen(
        TE = study_est,
        seTE = study_se,
        pval = study_p,
        lower = study_lower,
        upper = study_uppper
      )
      res <- summary(gen)
      Meta.estimate <- res$fixed$TE
      Meta.se <- res$fixed$seTE
      Meta.ciupper <- res$fixed$upper
      Meta.cilower <- res$fixed$lower
      Meta.p <- res$fixed$p
      Meta.I2 <- res$I2
      Meta.hetero.p <- res$pval.Q
      
      exposure_name = fea_name
      
      field_id = feature_selection[k,]$V2
      field_type = feature_selection[k,]$V1
      ivw_data <- rbind(ivw_data, c(exposure_name, field_id, field_type, id, 
                                    dis, tmp_res, 
                                    Meta.estimate, Meta.se, Meta.cilower, Meta.ciupper, 
                                    Meta.p, Meta.I2, Meta.hetero.p))
      
      print(paste(exposure_name, " finish"))
      
      j <- j+1
    }
    k <- k+1
    
  }
  
  i = 1
  ivw_file_columns = c("exposure", "exposure.field_id", "exposure.field_type", "exposure.id", "outcome")
  while (i <= length(disease_gwas_ids)){
    prefix = sprintf("mr%d_", i)
    cols = c(paste(prefix, "estimate", sep=""), paste(prefix, "cilower", sep=""), 
             paste(prefix, "ciupper", sep=""), paste(prefix, "se", sep=""), 
             paste(prefix, "pval", sep=""))
    ivw_file_columns = c(ivw_file_columns, cols)
    i = i+1
  }
  ivw_file_columns = c(ivw_file_columns, "meta.estimate", "meta.se", "meta.cilower", 
                       "meta.ciupper", "meta.p", "meta.I2", "meta.hetero.p")
  names(ivw_data) <- ivw_file_columns
  
  write.csv(ivw_data, file = sprintf('./feature_selection/%s/%s_ivw.csv', dis, phe), row.names = FALSE)
  
}

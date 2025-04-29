external_GWAS_process_1 <- function(snp_dis_df, 
                                    SNPs, 
                                    snp_col = "rsid",
                                    beta_col = "effect",
                                    se_col = "se",
                                    effect_allele_col = "a1",
                                    other_allele_col = "a2",
                                    eaf_col = "eaf",
                                    pval_col = "p",
                                    samplesize_col = "n",
                                    outcome = "outcome"){
  snp_dis_df <- snp_dis_df[snp_dis_df[,snp_col] %in% SNPs,]
  dis_df <- data.frame(
    SNP = snp_dis_df[,snp_col],
    beta = snp_dis_df[,beta_col],
    se = snp_dis_df[,se_col],
    effect_allele = snp_dis_df[,effect_allele_col],
    other_allele = snp_dis_df[,other_allele_col],
    eaf = snp_dis_df[,eaf_col],
    samplesize = snp_dis_df[,samplesize_col],
    pval = snp_dis_df[,pval_col]
  )
  dis_df <- format_data(dis_df, type="outcome")
  dis_df$outcome = outcome
  # tmp_dis_df = dis_df[dis_df$pval.outcome<0.05,]
  # if (nrow(tmp_dis_df) > 0){
  #   return (tmp_dis_df)
  # }
  return (dis_df)
}

external_GWAS_process_2 <- function(snp_dis_df,
                                    chr.postion,
                                    SNPs, 
                                    chr_pos_col = "chr.position",
                                    beta_col = "effect",
                                    se_col = "se",
                                    effect_allele_col = "a1",
                                    other_allele_col = "a2",
                                    eaf_col = "eaf",
                                    pval_col = "p",
                                    samplesize_col = "n",
                                    outcome = "outcome",
                                    is_eaf = T){
  snp_dis_df <- snp_dis_df[snp_dis_df[,chr_pos_col] %in% chr.postion,]
  k <- 1
  snp_dis_df$snp = NA
  while (k <= nrow(snp_dis_df)){
    snp_dis_df[k,]$snp <- SNPs[chr.postion == snp_dis_df[k,][,chr_pos_col]]
    k <- k+1
  }
  if (is_eaf) {
    dis_df <- data.frame(
      SNP = snp_dis_df$snp,
      beta = snp_dis_df[,beta_col],
      se = snp_dis_df[,se_col],
      effect_allele = snp_dis_df[,effect_allele_col],
      other_allele = snp_dis_df[,other_allele_col],
      eaf = snp_dis_df[,eaf_col],
      samplesize = snp_dis_df[,samplesize_col],
      pval = snp_dis_df[,pval_col]
    )
  } else {
    dis_df <- data.frame(
      SNP = snp_dis_df$snp,
      beta = snp_dis_df[,beta_col],
      se = snp_dis_df[,se_col],
      effect_allele = snp_dis_df[,effect_allele_col],
      other_allele = snp_dis_df[,other_allele_col],
      samplesize = snp_dis_df[,samplesize_col],
      pval = snp_dis_df[,pval_col]
    )
  }
  dis_df <- format_data(dis_df, type="outcome")
  dis_df$outcome = outcome
  # tmp_dis_df = dis_df[dis_df$pval.outcome<0.05,]
  # if (nrow(tmp_dis_df) > 0){
  #   return (tmp_dis_df)
  # }
  return (dis_df)
}

read_gwas_file <- function(file_path,
                           format_type,
                           pheno_name,
                           p_threshold = 1e-5) {
  source_python("./assist_gwas_process.py")
  
  phe_assoc <- extract_snps_from_gwas_by_p(file_path, p_threshold)
  if (is.null(phe_assoc)) {
    return (NULL)
  }
  alleles = c('A', 'T', 'C', 'G')
  phe_assoc <- phe_assoc[phe_assoc$A1 %in% alleles & phe_assoc$A2 %in% alleles,]
  
  # wrong_alleles_pair = c("GA", "AG", "CT", "TC", "AA", "TT", "CC", "GG")
  # alleles_pair = paste0(phe_assoc$A1, phe_assoc$A2)
  # phe_assoc <- phe_assoc[!(alleles_pair %in% wrong_alleles_pair),]
  
  phe_df <- data.frame(
    SNP = phe_assoc$SNP,
    beta = if ("BETA" %in% colnames(phe_assoc)) phe_assoc$BETA else log(phe_assoc$OR),
    se = phe_assoc$SE,
    effect_allele = phe_assoc$A1,
    chr = phe_assoc$CHR,
    pval = phe_assoc$P,
    phenotype = pheno_name,
    other_allele = phe_assoc$A2,
    eaf = phe_assoc$MAF
  )
  snps_pheno <- format_data(phe_df, type=format_type)
  snps_pheno[format_type] <- pheno_name
  return (snps_pheno)
}


clump <- function (dat, clump_kb = 10000, clump_r2 = 0.001, clump_p1 = 1, 
                   clump_p2 = 1, pop = "EUR", bfile = NULL, plink_bin = NULL) 
{
  pval_column <- "pval.exposure"
  if (!is.data.frame(dat)) {
    stop("Expecting data frame returned from format_data")
  }
  if ("pval.exposure" %in% names(dat) & "pval.outcome" %in% 
      names(dat)) {
    message("pval.exposure and pval.outcome columns present. Using pval.exposure for clumping.")
  }
  else if (!"pval.exposure" %in% names(dat) & "pval.outcome" %in% 
           names(dat)) {
    message("pval.exposure column not present, using pval.outcome column for clumping.")
    pval_column <- "pval.outcome"
  }
  else if (!"pval.exposure" %in% names(dat)) {
    message("pval.exposure not present, setting clumping p-value to 0.99 for all variants")
    dat$pval.exposure <- 0.99
  }
  else {
    pval_column <- "pval.exposure"
  }
  if (!"id.exposure" %in% names(dat)) {
    dat$id.exposure <- random_string(1)
  }
  d <- data.frame(rsid = dat$SNP, pval = dat[[pval_column]], 
                  id = dat$id.exposure)
  out <- ieugwasr::ld_clump(d, clump_kb = clump_kb, clump_r2 = clump_r2, 
                            clump_p = clump_p1, pop = pop, bfile = bfile, plink_bin = plink_bin)
  keep <- paste(dat$SNP, dat$id.exposure) %in% paste(out$rsid, 
                                                     out$id)
  return(dat[keep, ])
}

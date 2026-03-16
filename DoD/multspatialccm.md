R Package ‘multispatialCCM’


---



```r
library(multispatialCCM)
#See function for details - A is causally forced by B,
#but the reverse is not true.
ccm_data_out<-make_ccm_data()
Accm<-ccm_data_out$Accm
Bccm<-ccm_data_out$Bccm
#Calculate optimal E
maxE<-5 #Maximum E to test
#Matrix for storing output
Emat<-matrix(nrow=maxE-1, ncol=2); colnames(Emat)<-c("A", "B")
#Loop over potential E values and calculate predictive ability
#of each process for its own dynamics
for(E in 2:maxE) {
 #Uses defaults of looking forward one prediction step (predstep)
  #And using time lag intervals of one time step (tau)
  Emat[E-1,"A"]<-SSR_pred_boot(A=Accm, E=E, predstep=1, tau=1)$rho
  Emat[E-1,"B"]<-SSR_pred_boot(A=Bccm, E=E, predstep=1, tau=1)$rho
}
#Look at plots to find E for each process at which
#predictive ability rho is maximized
matplot(2:maxE, Emat, type="l", col=1:2, lty=1:2,
        xlab="E", ylab="rho", lwd=2)
legend("bottomleft", c("A", "B"), lty=1:2, col=1:2, lwd=2, bty="n")
#Results will vary depending on simulation.
#Using the seed we provide,
#maximum E for A should be 2, and maximum E for B should be 3.
#For the analyses in the paper, we use E=2 for all simulations.
E_A<-2
E_B<-3
#Check data for nonlinear signal that is not dominated by noise
#Checks whether predictive ability of processes declines with
#increasing time distance
#See manuscript and R code for details
signal_A_out<-SSR_check_signal(A=Accm, E=E_A, tau=1,predsteplist=1:10)
signal_B_out<-SSR_check_signal(A=Bccm, E=E_B, tau=1,predsteplist=1:10)
#Run the CCM test
#E_A and E_B are the embedding dimensions for A and B.
#tau is the length of time steps used (default is 1)
#iterations is the number of bootsrap iterations (default 100)
# Does A "cause" B?
#Note - increase iterations to 100 for consistant results
CCM_boot_A<-CCM_boot(Accm, Bccm, E_A, tau=1, iterations=100)
# Does B "cause" A?
CCM_boot_B<-CCM_boot(Bccm, Accm, E_B, tau=1, iterations=100)
#Test for significant causal signal
#See R function for details
(CCM_significance_test<-ccmtest(CCM_boot_A, CCM_boot_B))
#Plot results
plotxlimits<-range(c(CCM_boot_A$Lobs, CCM_boot_B$Lobs))
#Plot "A causes B"
plot(CCM_boot_A$Lobs, CCM_boot_A$rho, type="l", col=1, lwd=2,
xlim=c(plotxlimits[1], plotxlimits[2]), ylim=c(0,1),
xlab="L", ylab="rho")
#Add +/- 1 standard error
matlines(CCM_boot_A$Lobs,cbind(CCM_boot_A$rho-CCM_boot_A$sdevrho,
CCM_boot_A$rho+CCM_boot_A$sdevrho),lty=3, col=1)
#Plot "B causes A"
lines(CCM_boot_B$Lobs, CCM_boot_B$rho, type="l", col=2, lty=2, lwd=2)
#Add +/- 1 standard error
matlines(CCM_boot_B$Lobs,cbind(CCM_boot_B$rho-CCM_boot_B$sdevrho,
CCM_boot_B$rho+CCM_boot_B$sdevrho),lty=3, col=2)
legend("topleft",c("A causes B", "B causes A"),lty=c(1,2), col=c(1,2), lwd=2, bty="n")


```

OUT:

```
> Emat
          A       B
[1,] 0.4109265 0.4985892
[2,] 0.3872952 0.5589368
[3,] 0.3696862 0.5338485
[4,] 0.2503559 0.4702056


> # Does B "cause" A?
> CCM_boot_B<-CCM_boot(Bccm, Accm, E_B, tau=1, iterations=100)
> (CCM_significance_test<-ccmtest(CCM_boot_A,CCM_boot_B))
pval_a_cause_b pval_b_cause_a
      0.2         0.0



```


Olá! É excelente ver você explorando fronteiras mais avançadas da análise de séries temporais não-lineares. O Convergent Cross Mapping (CCM), introduzido por Sugihara e colaboradores, é uma ferramenta fantástica para sistemas dinâmicos complexos onde a tradicional Causalidade de Granger falha (especialmente quando há acoplamento não-linear).

Como você está buscando entender o mecanismo fundamental por trás desse resultado,  visualizar a reconstrução das variedades (manifolds) pelo Teorema de Takens ajuda a consolidar o conceito de que, em sistemas acoplados, o histórico de uma variável carrega a "impressão digital" da variável que a causou.

Vamos destrinchar os resultados do seu código R passo a passo:

### 1. Seleção da Dimensão de Imersão (A Matriz `Emat`)

Antes de rodar o CCM, o código busca a dimensão ótima (E) para reconstruir o espaço de fase de cada série isoladamente. A matriz `Emat` mostra o coeficiente de correlação (`rho`) entre os valores observados e previstos usando Simplex Projection para diferentes valores de E (variando de 2 a 5).

* **Para a série A:** O valor máximo de `rho` é **0.4109**, que ocorre na primeira linha da matriz. Como o loop começa em E=2, essa primeira linha corresponde a **E = 2**.
* **Para a série B:** O valor máximo de `rho` é **0.5589**, na segunda linha, o que corresponde a **E = 3**.

Isso explica por que o código define `E_A<-2` e `E_B<-3` logo em seguida. No seu segundo gráfico (E vs rho), vemos exatamente isso: a linha preta (A) tem um pico em E=2, e a linha vermelha tracejada (B) tem o pico em E=3.

### 2. O Teste de Causalidade CCM (Os P-Valores)

Aqui está a parte do CCM que muitas vezes confunde os analistas na primeira leitura: a lógica da previsão cruzada é invertida. **Para provar que B causa A, nós usamos o histórico da série A para tentar estimar os estados passados de B.** Se B causou A, a dinâmica de B está "codificada" em A.

Vamos olhar os resultados do seu `ccmtest`:

* **`pval_a_cause_b = 0.2`:** Este p-valor (20%) é superior ao nível de significância padrão de 5%. Isso significa que não há evidências estatísticas para afirmar que a série B consegue prever os estados de A no manifold cruzado. Logo, **A não causa B**.
* **`pval_b_cause_a = 0.0`:** Este p-valor é altamente significante. Significa que a série A consegue prever de forma excelente os estados de B. Seguindo a regra do CCM, isso prova que a informação fluiu de B para A. Logo, **B causa A**.

Isso confirma exatamente a premissa geradora dos seus dados simulados: *"A is causally forced by B, but the reverse is not true."*

### 3. A Análise Visual (Gráfico de Convergência)

O primeiro gráfico que você gerou é a prova visual dessa causalidade. A principal característica do CCM é a **convergência** — a habilidade preditiva (`rho`) deve aumentar e se estabilizar à medida que fornecemos mais dados históricos (aumentando o tamanho da biblioteca, `L`).

* A **linha tracejada vermelha ("B causes A")** mostra o `rho` subindo progressivamente de cerca de 0.1 para estabilizar acima de 0.6 conforme `L` aumenta em direção a 200. Essa convergência é a assinatura clássica de causalidade no CCM.
* A **linha sólida preta ("A causes B")** permanece plana e próxima a 0.1 (frequentemente cruzando os limites de erro em torno de zero), indicando que, por mais dados (L) que você adicione, não há ganho preditivo. Não há causalidade aqui.

### Próximos Passos e Recomendações

Sua análise inicial está perfeita e as conclusões são sólidas. Como seu parceiro de econometria, recomendo os seguintes testes de robustez antes de batermos o martelo (especialmente se formos aplicar isso a dados reais depois):

* **Análise do Time Lag (`tau`):** O código usou o padrão `tau = 1`. Em dados de alta frequência ou com inércia muito forte, um `tau` de 1 pode não desdobrar o atrator adequadamente (sucessivas observações seriam muito colineares). Plotar a Função de Autocorrelação (ACF) ou Informação Mútua (Mutual Information) das séries ajudaria a confirmar se `tau = 1` é o ideal.
* **Testes com Surrogate Data:** Embora o bootstrap do pacote seja bom, sistemas reais com forte sazonalidade ou ciclos de baixa frequência podem gerar falsos positivos ("spurious cross-mapping"). Testar contra dados substitutos (como *Twin Surrogates* ou *Ebisuzaki phase-randomization*) garante que a causalidade detectada se deve à dinâmica não-linear e não a propriedades lineares genéricas.

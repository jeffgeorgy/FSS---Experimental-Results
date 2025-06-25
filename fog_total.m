%% Limpa a memoria e carrega os dados
clear % Limpa a memoria
clc % Limpa a janela de comando
close all
% Carrega os dados da planta
load('Fogtrein.txt'); % Carrega dados do foguete
u = Fogtrein(:,1:2);
y = Fogtrein(:,3);
centros1 = [11.77,21.45,41.02,41.18,29.37,15.69,45.95,0,0;85.04,82.16,74.78,46.52,36.36,31.68,62.34,90.41,30];
centros2 = [6.44,1.30,358.92,347.84,335.3,323.02,303.29;90.09,80.98,79.26,69.02,55.25,43.19,30.86];
%% Inicializacao do Algoritmo
% Parametros do algoritmo
r = 0.18; % raio de influencia de um cluster
m = 1.4; %Ponderamento exponencial para calculo de pertinencias
gama = 0.22; % Ponderamento das informacoes no potencial
tic % Inicializa a contagem de tempo
% Variaveis do algoritmo
k = 1; % Amostra inicial
R(1) = 1; % Numero de regras inicial
x = [u';y']; % Vetor de dados
n = length(x(:,1)); % Numero de coordenadas
for i = 1:length(x(:,1))
    mx = max(x(i,:));
    mn = min(x(i,:));
    if mx == mn
        break
    else
        xn(i,:) = (x(i,:)-mn)/(mx-mn);
    end
end
xcn{1} = [xn(:,1)]; % Primeiro centro de cluster normalizado
N = length(x(1,:)); % Numero de amostras
Upsilon_c{1}(1) = min([r^2,1]); % Taxa de variacao do primeiro centro
Pc{1}(1) = 1; % Potencial do primeiro centro de cluster
NPC{1}(1) = 0; % Numero de dados do potencial do primeiro centro
psi(1) = 1; % indice desde a ultima criacao de cluster
max_Upsilon(1) = 0; % Valor maximo da taxa de variacao para normalizacao
NF{1}(:,:,1) = eye(n); % Inicializacao do numerador da inversa da matriz de covariancia nebulosa
DF{1}(1) = 1; % Inicializacao do denominador da matriz de covariancia nebulosa
F{1}(:,:,1) = NF{1}(:,:,1)/DF{1}(1);  % Inicializacao da inversa matriz de covariancia nebulosa
S{1}(1) = 1; % Suporte do primeiro cluster
DM{1}(1) = 0; % Distancia media amostral do primeiro cluster
epsilon{1}(1) = r; % Raio do primeiro cluster
S_{1}(1) = 1; % Suporte normalizado do primeiro cluster
s{1}(1) = Pc{k}(1)*exp(-S_{1}(1)); % Fator de sensibilidade do primeiro cluster
deslocamento{1} = xcn{1}(:,1); % Para plotar
mu(1,1) = 1;
muf = mu;
DA{1} = x(:,1);
centros_processo = xcn{1};
%% Parametros ODIK/ERA
Ni = 500; % Instante de inicializacao
nent = length(u(1,:));
nsai = length(y(1,:));
p_pm = 1;
v = [u';y'];
alphaf = 10;
betaf = 10;
lambda = 1;
fat_alt = 0.01;
%% Realizacao do Algoritmo
for k = 2:N
    k % Mostra o valor de k
    R(k,1) = R(k-1);
    psi(k,1) = psi(k-1);
    xcn{k,1} = xcn{k-1};
    Upsilon_c{k,1} = Upsilon_c{k-1};
    s{k,1} = s{k-1};
    epsilon{k,1} = epsilon{k-1};
    Upsilon(k,1) = norm(xn(:,k) - xn(:,k-1)); % Calcula a norma da primeira diferenca (taxa de variacao)
    if Upsilon(k) > max_Upsilon(k-1) % Se valor for maior que o maximo
        max_Upsilon(k,1) = Upsilon(k); % Atualiza o maximo
    else
        max_Upsilon(k,1) = max_Upsilon(k-1); % Mantem o mesmo valor maximo
    end
    if max_Upsilon(k) > 0
        Upsilon_(k,1) = Upsilon(k)/max_Upsilon(k); % Normaliza a taxa de variacao
    else
        Upsilon_(k,1) = Upsilon(k);
    end
    for i = 1:R(k)  % Para todos os centros de clusters
        dist = sqrt((xn(:,k)-xcn{k}(:,i))'*det(F{k-1}(:,:,i))^(1/n)*inv(F{k-1}(:,:,i))*...
            (xn(:,k)-xcn{k}(:,i))); % Norma induzida
        if isnan(dist) % Para verificacao de erros
            dist
            dist;
        end
        D_(i,k) = exp(dist)-1; % Medicao da distancia adaptativa do dado atual para os centros
    end
    P(k,1) = gama*exp(-4/r^2*Upsilon_(k)^2); % Inicializa o potencial da amostra atual
    eta(k,1) = k-psi(k)-1; % Numero de amostras desde a ultima criacao de cluster
    for j = (psi(k)+1):(k-1) % Para todos os eta(k) dados desde a ultima criacao de cluster
        P(k) = P(k) + 1/(R(k)+eta(k))*(1-gama)*exp(-4/r^2*norm(xn(:,k)...
            -xn(:,j))^2); % Calcula o potencial do dado atual
    end
    for j = 1:R(k) % Para todos os centros de clusters existentes
        P(k,1) = P(k) + 1/(R(k)+eta(k))*(1-gama)*exp(-4/r^2*D_(j,k)^2); % Calcula o potencial do dado atual
        NPC{k}(j,1) = NPC{k-1}(j) + 1; % Atualiza o numero de amostras usadas nos potenciais dos centros
        Pc{k,1}(j,1) = (NPC{k}(j)-1)/NPC{k}(j)*Pc{k-1}(j) + 1/NPC{k}(j)*...
            ((1-gama)*exp(-4/r^2*D_(j,k)^2) + gama*exp(-4/r^2*...
            Upsilon_c{k}(j)^2)); % Atualiza potenciais de centros
    end
    dmin = 1e50; % Inicializa a distancia minima
    for i = 1:R(k) % Para todos os centros de clusters
        if D_(i,k) < dmin  % Se a distancia for menor que a minima
            dmin = D_(i,k); % Atualiza distancia minima
            indc_prox = i; % Indice do cluster mais proximo
        end
    end
    
    if P(k) > (1-s{k}(indc_prox))*Pc{k}(indc_prox) % Condicao para selecao do dado atual
        if dmin < epsilon{k}(indc_prox) % Condicao de proximidade
            rou = P(k)/(P(k)+Pc{k}(indc_prox)); % Peso do dado atual no cruzamento
            xcn{k,1}(:,indc_prox) = rou*xn(:,k) + (1-rou)*xcn{k}...
                (:,indc_prox); % Atualizacao do centro de cluster normalizado
            deslocamento{indc_prox,1} = [deslocamento{indc_prox},xcn{k}(:,indc_prox)];
            centros_processo = [centros_processo,xcn{k}(:,indc_prox)];
            Pc{k}(indc_prox) = rou*P(k)+(1-rou)*Pc{k}(indc_prox); % Atualizacao do potencial do centro
            NPC{k}(indc_prox) = ceil(rou*(R(k)+eta(k))+(1-rou)*NPC{k}...
                (indc_prox)); % Atualizacao do numero NPC
            Upsilon_c{k,1}(indc_prox,1) = rou*Upsilon_(k) + (1-rou)*...
                Upsilon_c{k}(indc_prox); % Atualizacao da taxa de variacao do centro
            dist = sqrt((xn(:,k)-xcn{k}(:,indc_prox))'*det(F{k-1}...
                (:,:,indc_prox))^(1/n)*inv(F{k-1}(:,:,indc_prox))*...
                (xn(:,k)-xcn{k}(:,indc_prox))); % Computa a norma induzida
            D_(indc_prox,k) = exp(dist)-1; % Atualizacao da distancia do dado ao centro
        else % Condicao para criacao de cluster
            R(k,1) = R(k) + 1; % Incrementa o numero de regras
            DA{R(k),1} = [];
            S{k-1}(R(k),1) = 0; % Inicializa o suporte do novo cluster
            DM{k-1}(R(k),1) = 0; % Inicializa a distancia media amostral do novo cluster
            epsilon{k}(R(k),1) = r; % Raio do novo cluster
            S_{k-1}(R(k),1) = 0; % Suporte normalizado do novo cluster
            xcn{k,1}(:,R(k)) = xn(:,k); % Centro normalizado do novo cluster
            deslocamento{R(k),1} = xcn{k}(:,R(k));
            centros_processo = [centros_processo,xcn{k,1}(:,R(k))];
            Pc{k}(R(k),1) = P(k); % Potencial do novo centro
            s{k}(R(k),1) = Pc{k}(R(k))*exp(-S_{k-1}(R(k))); % Fator de sensibilidade do novo cluster
            NPC{k}(R(k),1) = R(k) - 1 + eta(k); % Numero de dados usados para medicao do potencial do novo cluster
            Upsilon_c{k,1}(R(k),1) = Upsilon_(k); % Taxa de variacao do novo centro
            NF{k-1}(:,:,R(k)) = eye(n); % Numerador da inversa da nova matriz de covariancia nebulosa
            DF{k-1}(R(k),1) = 1; % Denominadorda nova matriz de covariancia nebulosa
            F{k-1}(:,:,R(k)) = NF{k-1}(:,:,R(k))/DF{k-1}(R(k)); % Inversa da matriz de covariancia nebulosa
            D_(R(k),k) = 0; % Distancia entre o dado atual e o novo cluster
            psi(k,1) = k; % Atualiza o indice do ultimo centro criado para o atual
            for i = 1:R(k)-1 % Para todos os clusters anteriores
                DM{k-1}(i,1) = (R(k-1))/R(k)*DM{k-1}(i); % Reduz a distancia media amostral
            end
            if k > Ni % Inicializa parametros do consequente
                P_cov{R(k),1} = 0;
                Y_m{R(k),1} = 0;
                for i = 1:(R(k)-1)
                    P_cov{R(k)} = P_cov{R(k)} + 1/(R(k)-1)*P_cov{i};
                    Y_m{R(k)} = Y_m{R(k)} + 1/(R(k)-1)*Y_m{i};
                end
                Z{R(k),1} = 0;
                Y_0{R(k),1} = 0;
                Y_{R(k),1} = 0;
                Y_1{R(k),1} = 0;
                Y_2{R(k),1} = 0;
                Y{R(k),1} = 0;
                H0{R(k),1} = 0;
                H1{R(k),1} = 0;
                R_svd{R(k),1} = 0;
                S_svd{R(k),1} = 0;
                Sigma{R(k),1} = 0;
                Sigma_n{R(k),1} = 0;
                Rn{R(k),1} = 0;
                Sn{R(k),1} = 0;
                nf(R(k),1) = ceil(mean(nf(1:R(k)-1)));
                A_m{R(k),k-1} = zeros(nf(R(k)),nf(R(k)));
                B_m{R(k),k-1} = zeros(nf(R(k)),nent);
                C_m{R(k),k-1} = zeros(nsai,nf(R(k)));
                D_m{R(k),k-1} = zeros(nsai,nent);
                Yo{R(k),1} = 0;
                G{R(k),1} = 0;
                Po{R(k),1} = 0;
                Yom{R(k),1} = 0;
                xz{R(k),1}(:,k-1) = zeros(nf(R(k)),1);
                yer{R(k),1}(k-1,:) = zeros(1,nsai);
            end
        end
    end
    if R(k) > 1 % Se ha mais de um cluster
        fimcrossover = 0; % Detecta se nao e mais necessario cruzamentos
        while ~fimcrossover % Enquanto for necessario cruzamentos
            houvecrossover = 0; % Detecta se houve cruzamento
            Dc{k,1} = zeros(R(k),R(k)); % Inicializa as distâncias entre centros
            for i = 1:R(k) % Para todos os clusters
                for j = 1:R(k) % Para todos os clusters
                    dist = sqrt((xcn{k}(:,i)-xcn{k}(:,j))'*det(F{k-1}(:,:,i))^(1/n)*inv(F{k-1}(:,:,i))*...
                        (xcn{k}(:,i)-xcn{k}(:,j))); % Computa a norma induzida
                    Dc{k,1}(i,j) = exp(dist)-1; % Computa a distancia entre os centros
                end
            end
            for i = 1:R(k) % Para todos os clusters
                for j = 1:R(k) % Para todos os clusters
                    i_cp = [i,j]; % Vetor de indices para cruzamento
                    if Dc{k,1}(i_cp(1),i_cp(2)) < epsilon{k}(i_cp(1))...
                            && Dc{k,1}(i_cp(2),i_cp(1)) < ...
                            epsilon{k}(i_cp(2)) && i~=j % Se tiver sobreposicao mútua de clusters distintos
                        houvecrossover = 1; % Detecta que houve cruzamento
                        rou = Pc{k}(i_cp(1))/(Pc{k}(i_cp(1))+...
                            Pc{k}(i_cp(2))); % Peso do primeiro cluster no cruzamento
                        xcn{k}(:,R(k)+1) = rou*xcn{k}(:,i_cp(1)) +...
                            (1-rou)*xcn{k}(:,i_cp(2)); % Centro de cluster normalizado resultante
                        deslocamento{R(k)+1,1} = xcn{k}(:,R(k)+1);
                        centros_processo = [centros_processo,xcn{k,1}(:,R(k)+1)];
                        Pc{k}(R(k)+1,1) = rou*Pc{k}(i_cp(1))+(1-rou)...
                            *Pc{k}(i_cp(2)); % Potencial do centro de cluster resultante
                        NPC{k}(R(k)+1,1) = ceil(rou*NPC{k}(i_cp(1))+...
                            (1-rou)*NPC{k}(i_cp(2))); % Numero NPC do centro de cluster resultante
                        Upsilon_c{k}(R(k)+1,1) = rou*Upsilon_c{k}(i_cp(1))...
                            + (1-rou)*Upsilon_c{k}(i_cp(2)); % Taxa de variacao do centro de cluster resultante
                        NF{k-1}(:,:,R(k)+1) = rou*NF{k-1}(:,:,i_cp(1))+(1-rou)*NF{k-1}(:,:,i_cp(2)); % Inversa do numerador da MCN do cluster resultante
                        DF{k-1}(R(k)+1) = rou*DF{k-1}(i_cp(1)) +  (1-rou)*DF{k-1}(i_cp(2)); % Denominador da MCN do cluster resultante
                        F{k-1}(:,:,R(k)+1) = NF{k-1}...
                            (:,:,R(k)+1)/DF{k-1}(R(k)+1); % Inversa da MCN do cluster resultante
                        DA{R(k)+1,1} = [DA{i_cp(1)},DA{i_cp(2)}];
                        S{k-1}(R(k)+1,1) = S{k-1}(i_cp(1)) + S{k-1}(i_cp(2)); % Suporte do cluster resultante
                        S_{k-1}(R(k)+1,1) = S{k-1}(R(k)+1)/max(S{k-1}); % Suporte normalizado do cluster resultante
                        DM{k-1}(R(k)+1,1) = DM{k-1}(i_cp(1)) +...
                            DM{k-1}(i_cp(2)); % Distancia media amostral do cluster resultante
                        epsilon{k}(R(k)+1,1) = DM{k-1}(R(k)+1) +...
                            (r-DM{k-1}(R(k)+1))/...
                            (S{k-1}(R(k)+1)^(1/(2*n))); % Raio do cluster resultante
                        s{k}(R(k)+1,1) = Pc{k}(R(k)+1)*exp(-S_{k-1}...
                            (R(k)+1)); % Fator de sensibilidade do cluster resultante
                        dist = sqrt((xn(:,k)-xcn{k}(:,R(k)+1))'*...
                            det(F{k-1}(:,:,R(k)+1))^(1/n)*inv(F{k-1}...
                            (:,:,R(k)+1))*(xn(:,k)-xcn{k}(:,R(k)+1))); % Distancia de norma induzida
                        D_(R(k)+1,k) = exp(dist)-1; % Distancia do dado atual ao cluster resultante
                        mu(R(k)+1,:) = zeros(size(mu(1,:))); % Inicializa pertinencia do dado atual ao cluster resultante
                        % Remove todas as variaveis dos clusters mesclados
                        xcn{k}(:,max(i_cp)) = [];
                        xcn{k}(:,min(i_cp)) = [];
                        deslocamento(max(i_cp)) = [];
                        deslocamento(min(i_cp)) = [];
                        Pc{k}(max(i_cp)) = [];
                        Pc{k}(min(i_cp)) = [];
                        NPC{k}(max(i_cp)) = [];
                        NPC{k}(min(i_cp)) = [];
                        Upsilon_c{k}(max(i_cp)) = [];
                        Upsilon_c{k}(min(i_cp)) = [];
                        NF{k-1}(:,:,max(i_cp)) = [];
                        NF{k-1}(:,:,min(i_cp)) = [];
                        DF{k-1}(max(i_cp)) = [];
                        DF{k-1}(min(i_cp)) = [];
                        F{k-1}(:,:,max(i_cp)) = [];
                        F{k-1}(:,:,min(i_cp)) = [];
                        DA(max(i_cp)) = [];
                        DA(min(i_cp)) = [];
                        S{k-1}(max(i_cp)) = [];
                        S{k-1}(min(i_cp)) = [];
                        DM{k-1}(max(i_cp)) = [];
                        DM{k-1}(min(i_cp)) = [];
                        epsilon{k}(max(i_cp)) = [];
                        epsilon{k}(min(i_cp)) = [];
                        S_{k-1}(max(i_cp)) = [];
                        S_{k-1}(min(i_cp)) = [];
                        s{k}(max(i_cp)) = [];
                        s{k}(min(i_cp)) = [];
                        D_(max(i_cp),:) = [];
                        D_(min(i_cp),:) = [];
                        mu(max(i_cp),:) = [];
                        mu(min(i_cp),:) = [];
                        if k > Ni % Parametros consequente
                            nf(R(k)+1,1) = rou*nf(i_cp(1)) + (1-rou)*nf(i_cp(2));
                            A_m{R(k)+1,k-1} = zeros(nf(R(k)+1),nf(R(k)+1));
                            B_m{R(k)+1,k-1} = zeros(nf(R(k)+1),nent);
                            C_m{R(k)+1,k-1} = zeros(nsai,nf(R(k)+1));
                            D_m{R(k)+1,k-1} = zeros(nsai,nent);
                            P_cov{R(k)+1,1} = rou*P_cov{i_cp(1)} + (1-rou)*P_cov{i_cp(2)};
                            Y_m{R(k)+1,1} = rou*Y_m{i_cp(1)} + (1-rou)*Y_m{i_cp(2)};
                            Y_0{R(k)+1,1} = 0;
                            Y_{R(k)+1,1} = 0;
                            Y_1{R(k)+1,1} = 0;
                            Y_2{R(k)+1,1} = 0;
                            Y{R(k)+1,1} = 0;
                            H0{R(k)+1,1} = 0;
                            H1{R(k)+1,1} = 0;
                            R_svd{R(k)+1,1} = 0;
                            S_svd{R(k)+1,1} = 0;
                            Sigma{R(k)+1,1} = 0;
                            Sigma_n{R(k)+1,1} = 0;
                            Rn{R(k)+1,1} = 0;
                            Sn{R(k)+1,1} = 0;
                            Yo{R(k)+1,1} = 0;
                            G{R(k)+1,1} = 0;
                            Po{R(k)+1,1} = 0;
                            Yom{R(k)+1,1} = 0;
                            xz{R(k)+1}(:,k-1) = zeros(nf(R(k)+1),1);
                            if k > Ni+1
                                Z{R(k)+1,1} = rou*Z{i_cp(1)} + (1-rou)*Z{i_cp(2)};
                                yer{R(k)+1,1}(k-1,:) = zeros(1,nsai);
                                Z(max(i_cp)) = [];
                                Z(min(i_cp)) = [];
                                yer(max(i_cp)) = [];
                                yer(min(i_cp)) = [];
                            end
                            P_cov(max(i_cp)) = [];
                            P_cov(min(i_cp)) = [];
                            Y_m(max(i_cp)) = [];
                            Y_m(min(i_cp)) = [];
                            Y_0(max(i_cp)) = [];
                            Y_0(min(i_cp)) = [];
                            Y_(max(i_cp),:) = [];
                            Y_(min(i_cp),:) = [];
                            Y_1(max(i_cp),:) = [];
                            Y_1(min(i_cp),:) = [];
                            Y_2(max(i_cp),:) = [];
                            Y_2(min(i_cp),:) = [];
                            Y(max(i_cp),:) = [];
                            Y(min(i_cp),:) = [];
                            H0(max(i_cp)) = [];
                            H0(min(i_cp)) = [];
                            H1(max(i_cp)) = [];
                            H1(min(i_cp)) = [];
                            R_svd(max(i_cp)) = [];
                            R_svd(min(i_cp)) = [];
                            S_svd(max(i_cp)) = [];
                            S_svd(min(i_cp)) = [];
                            Sigma(max(i_cp)) = [];
                            Sigma(min(i_cp)) = [];
                            Sigma_n(max(i_cp)) = [];
                            Sigma_n(min(i_cp)) = [];
                            Rn(max(i_cp)) = [];
                            Rn(min(i_cp)) = [];
                            Sn(max(i_cp)) = [];
                            Sn(min(i_cp)) = [];
                            A_m(max(i_cp),:) = [];
                            A_m(min(i_cp),:) = [];
                            B_m(max(i_cp),:) = [];
                            B_m(min(i_cp),:) = [];
                            C_m(max(i_cp),:) = [];
                            C_m(min(i_cp),:) = [];
                            D_m(max(i_cp),:) = [];
                            D_m(min(i_cp),:) = [];
                            Yo(max(i_cp),:) = [];
                            Yo(min(i_cp),:) = [];
                            G(max(i_cp),:) = [];
                            G(min(i_cp),:) = [];
                            Po(max(i_cp)) = [];
                            Po(min(i_cp)) = [];
                            Yom(max(i_cp)) = [];
                            Yom(min(i_cp)) = [];
                            xz(max(i_cp)) = [];
                            xz(min(i_cp)) = [];
                            nf(max(i_cp)) = [];
                            nf(min(i_cp)) = [];
                        end
                        R(k) = R(k)-1; % Diminui a quantidade de regras
                        break % Para o laco for interior se houve crossover
                    end
                end
                if houvecrossover % Se houve crossover
                    break % Para o laco for exterior
                end
            end
            if ~houvecrossover % Se nao houve crossover
                fimcrossover = 1; % Determina o fim do processo de crossover
            end
        end
    end
    dmin = 1e50; % Inicializa a distancia minima
    for i = 1:R(k) % Para todos os clusters
        if D_(i,k) < dmin % Se a distancia for menor que a minima
            dmin = D_(i,k); % Atualiza a distancia minima
            indc_prox = i; % Define o cluster mais proximo
        end
    end
    DA{indc_prox,1} = [DA{indc_prox},x(:,k)];
    S{k,1}(indc_prox,1) =  S{k-1}(indc_prox) + 1; % Atualiza o suporte do cluster mais proximo
    DM{k,1}(indc_prox,1) = (S{k}(indc_prox)-1)/S{k}(indc_prox)*DM{k-1}...
        (indc_prox) + 1/S{k}(indc_prox)*D_(indc_prox,k); % Atualiza a distancia media amostral do clusters mais proximo
    for i = 1:R(k) % Para todos os clusters
        if i ~= indc_prox % Se nao for o cluster mais proximo
            S{k}(i,1) = S{k-1}(i); % Mantem o valor do suporte
            DM{k}(i,1) = DM{k-1}(i); % Mantem o valor da distancia media amostral
        end
    end
    epsilon{k,1} = DM{k} + (r-DM{k})./(S{k}.^(1/(2*n))); % Atualiza o raio dos clusters
    for i = 1:R(k) % Para todos os clusters
        S_{k,1}(i,1) = S{k}(i)/max(S{k}); % Atualiza o suporte normalizado
        s{k,1}(i,1) = Pc{k}(i)*exp(-S_{k}(i)); % Atualiza o fator de sensibilidade
    end
    indcn = 0; % Inicializa o indice de cluster com distancia nula
    for i = 1:R(k) % Para todos os clusters
        if D_(i,k) == 0 % Se a distancia for nula
            indcn = i; % Detecta cluster com distancia nula
        end
    end
    if indcn == 0 % Se nao houver centro de cluster igual ao dado atual
        for i = 1:R(k) % Para todos os clusters
            if ~isinf(D_(i,k)) % Se a distancia nao for infinita
                mu(i,k) = 0; % Inicializa pertinencia
                for j = 1:R(k) % Para todos os clusters
                    mu(i,k) = mu(i,k) + (D_(i,k)/D_(j,k))^(2/(m-1)); % Calcula o inverso da pertinencia
                end
                mu(i,k) = 1/mu(i,k); % Calcula a pertinencia
            else % Se a distancia for infinita
                mu(i,k) = 0; % Define pertinencia nula
            end
            muf(i,k) = mu(i,k);
        end
    else % Se houver centro de cluster igual ao dado atual
        mu(indcn,k) = 1; % Define pertinencia unitaria para este cluster
        muf(indcn,k) = mu(indcn,k);
    end
    for i = 1:R(k) % Para todos os clusters
        NF{k,1}(:,:,i) = NF{k-1}(:,:,i) + mu(i,k)^m*(xn(:,k)-xcn{k}(:,i))*(xn(:,k)-xcn{k}(:,i))';
        DF{k,1}(i,1) = DF{k-1}(i) + mu(i,k)^(m); % Atualiza o denominador da MCN
        F{k,1}(:,:,i) = NF{k}(:,:,i)/DF{k}(i); % Atualiza a inversa da MCN
    end
    if k == Ni % Inicializa os parametros de Markov
        % Mede as pertinencias dos Ni primeiros dados em relacao aos
        % clusters formados
        mu(:,1:Ni) = zeros(R(k),Ni);
        for j = 1:Ni
            indcn = 0; % Inicializa o indice de cluster com distancia nula
            for i = 1:R(k) % Medicao da distancia adaptativa do dado atual para os centros
                dist = sqrt((xn(:,j)-xcn{k}(:,i))'*det(F{k}(:,:,i))^(1/n)*inv(F{k}(:,:,i))*...
                    (xn(:,j)-xcn{k}(:,i)));
                D_(i,j) = exp(dist)-1;
                if D_(i,j) == 0 % Se a distancia for nula
                    indcn = i; % Detecta cluster com distancia nula
                end
            end
            if indcn == 0 % Se nao houver centro de cluster igual ao dado atual
                for i = 1:R(k) % Para todos os clusters
                    if ~isinf(D_(i,j)) % Se a distancia nao for infinita
                        mu(i,j) = 0; % Inicializa pertinencia
                        for jj = 1:R(k) % Para todos os clusters
                            mu(i,j) = mu(i,j) + (D_(i,j)/D_(jj,j))^(2/(m-1)); % Calcula o inverso da pertinencia
                        end
                        mu(i,j) = 1/mu(i,j); % Calcula a pertinencia
                    else % Se a distancia for infinita
                        mu(i,j) = 0; % Define pertinencia nula
                    end
                end
            else % Se houver centro de cluster igual ao dado atual
                mu(indcn,j) = 1; % Define pertinencia unitaria para este cluster
            end
        end
        y_v = y((p_pm+1):Ni,:)';
        V = u((p_pm+1):Ni,:)';
        tau_a = p_pm;
        for i = 1:p_pm
            V = [V;v(:,(p_pm+1-i):(Ni-i))];
        end
        for i = 1:R(k)
            W{i,1} = diag(mu(i,(p_pm+1):Ni));
            P_cov{i,1} = (V*W{i}*V')^-1;
            Y_m{i,1} = y_v*W{i}*V'*P_cov{i};
            yves{i,1} = Y_m{i}*V;
            % for j = 1:nent
            %     figure
            %     plot(y_v(j,:))
            %     hold on
            %     plot(yves{i}(j,:),'--')
            % end
            Y_0{i,1} = Y_m{i}(1:nsai,1:nent);
            for j = 1:p_pm
                Y_{i,j} = Y_m{i}(1:nsai,(2+(j-1)*(nent+nsai)):(j*(nent+nsai)+1));
                Y_1{i,j} = Y_{i,j}(:,1:nent);
                Y_2{i,j} = -Y_{i,j}(:,(nent+1):(nsai+nent));
            end
            for j = 1:p_pm
                Y{i,j} = Y_1{i,j} - Y_2{i,j}*Y_0{i};
                for jj = 1:(j-1)
                    Y{i,j} = Y{i,j} - Y_2{i,jj}*Y{i,j-jj};
                end
            end
            for j = (p_pm+1):(alphaf+betaf)
                Y{i,j} = 0;
                for jj = 1:p_pm
                    Y{i,j} = Y{i,j} - Y_2{i,jj}*Y{i,j-jj};
                end
            end
            H0{i,1} = cell2mat(Y(i,1:betaf));
            H1{i,1} = cell2mat(Y(i,2:(betaf+1)));
            for j = 2:alphaf
                H0{i,1} = [H0{i};cell2mat(Y(i,j:(betaf+j-1)))];
                H1{i,1} = [H1{i};cell2mat(Y(i,(j+1):(betaf+j)))];
            end
            [R_svd{i,1},Sigma{i,1},S_svd{i,1}] = svd(H0{i});
            rank(H0{i})
            nf(i,1) = rank(H0{i});
            Sigma_n{i,1} = Sigma{i}(1:nf(i),1:nf(i));
            Rn{i,1} = R_svd{i}(:,1:nf(i));
            Sn{i,1} = S_svd{i}(:,1:nf(i));
            Er = [eye(nent);zeros((betaf-1)*nent,nent)];
            Em = [eye(nsai);zeros((alphaf-1)*nsai,nsai)];
            A_m{i,Ni} = Sigma_n{i}^(-1/2)*Rn{i}'*H1{i}*Sn{i}*Sigma_n{i}^(-1/2);
            B_m{i,Ni} = Sigma_n{i}^(1/2)*Sn{i}'*Er;
            C_m{i,Ni} = Em'*Rn{i}*Sigma_n{i}^(1/2);
            D_m{i,Ni} = Y_0{i};
            for kk = 1:p_pm
                Yo{i,kk} = Y_2{i,kk};
                for j = 1:(kk-1)
                    Yo{i,kk} = Yo{i,kk} - Y_2{i,j}*Yo{i,kk-j};
                end
            end
            for kk = (p_pm+1):(alphaf+betaf)
                Yo{i,kk} = 0;
                for j = 1:p_pm
                    Yo{i,kk} = Yo{i,kk} - Y_2{i,j}*Yo{i,kk-j};
                end
            end
            Po{i,1} = C_m{i,Ni};
            Yom{i,1} = Yo{i,1};
            for kk = 1:(alphaf+betaf-1)
                Po{i,1} = [Po{i};C_m{i,Ni}*A_m{i,Ni}^kk];
                Yom{i,1} = [Yom{i};Yo{i,kk+1}];
            end
            G{i,Ni} = (Po{i}'*Po{i})^-1*Po{i}'*Yom{i};
            % xz{i}(:,Ni-tau_a) = zeros(nf(i),1);
            % for j = (Ni-tau_a+1):Ni
            %     xz{i}(:,j) = A_m{i,Ni}*xz{i}(:,j-1)+B_m{i,Ni}*u(j-1,:)';
            % end
            xz{i,1}(:,Ni) = zeros(nf(i),1);
            % yer{i}(Ni,:) = (C_m{i,Ni}*xz{i}(:,Ni)+D_m{i,Ni}*u(Ni,:)')';
        end
        ye(Ni,:) = zeros(1,nsai);
        % for i = 1:R
        %     ye(Ni,:) = ye(Ni,:) + mu(i,Ni)*yer{i}(Ni,:);
        % end
        acionamentos = 0;
        k
    end
    if k > Ni
        ppi = u(k,:)';
        for i = 1:p_pm
            ppi = [ppi;v(:,k-i)];
        end
        ye(k,:) = zeros(1,nsai);
        for i = 1:R(k)
            Z{i,1} = ppi'*P_cov{i}/(lambda/mu(i,k)+ppi'*P_cov{i}*ppi);
            P_cov{i,1} = lambda^-1*P_cov{i}*(eye(length(P_cov{i}))-ppi*Z{i});
            Y_m{i,1} = Y_m{i} + (y(k,:)'-Y_m{i}*ppi)*Z{i};
            Y_0{i,1} = Y_m{i}(1:nsai,1:nent);
            for j = 1:p_pm
                Y_{i,j} = Y_m{i}(1:nsai,(j*nent+(j-1)*nsai+1):((j+1)*nent+j*nsai));
                Y_1{i,j} = Y_{i,j}(:,1:nent);
                Y_2{i,j} = -Y_{i,j}(:,(nent+1):(nsai+nent));
            end
            for j = 1:p_pm
                Y{i,j} = Y_1{i,j} - Y_2{i,j}*Y_0{i};
                for jj = 1:(j-1)
                    Y{i,j} = Y{i,j} - Y_2{i,jj}*Y{i,j-jj};
                end
            end
            for j = (p_pm+1):(alphaf+betaf)
                Y{i,j} = 0;
                for jj = 1:p_pm
                    Y{i,j} = Y{i,j} - Y_2{i,jj}*Y{i,j-jj};
                end
            end
            H0{i,1} = cell2mat(Y(i,1:betaf));
            H1{i,1} = cell2mat(Y(i,2:(betaf+1)));
            for j = 2:alphaf
                H0{i,1} = [H0{i};cell2mat(Y(i,j:(betaf+j-1)))];
                H1{i,1} = [H1{i};cell2mat(Y(i,(j+1):(betaf+j)))];
            end
            [R_svd{i,1},Sigma{i,1},S_svd{i,1}] = svd(H0{i});
            Sigma_n{i,1} = Sigma{i}(1:nf(i),1:nf(i));
            Rn{i,1} = R_svd{i}(:,1:nf(i));
            Sn{i,1} = S_svd{i}(:,1:nf(i));
            Er = [eye(nent);zeros((betaf-1)*nent,nent)];
            Em = [eye(nsai);zeros((alphaf-1)*nsai,nsai)];
            A_m{i,k} = Sigma_n{i}^(-1/2)*Rn{i}'*H1{i}*Sn{i}*Sigma_n{i}^(-1/2);
            B_m{i,k} = Sigma_n{i}^(1/2)*Sn{i}'*Er;
            C_m{i,k} = Em'*Rn{i}*Sigma_n{i}^(1/2);
            D_m{i,k} = Y_0{i};
            for ii = 1:p_pm
                Yo{i,ii} = Y_2{i,ii};
                for j = 1:(ii-1)
                    Yo{i,ii} = Yo{i,ii} - Y_2{i,j}*Yo{i,ii-j};
                end
            end
            for ii = (p_pm+1):(alphaf+betaf)
                Yo{i,ii} = 0;
                for j = 1:p_pm
                    Yo{i,ii} = Yo{i,ii} - Y_2{i,j}*Yo{i,ii-j};
                end
            end
            Po{i,1} = C_m{i,k};
            Yom{i,1} = Yo{i,1};
            for j = 1:(alphaf+betaf-1)
                Po{i,1} = [Po{i};C_m{i,k}*A_m{i,k}^j];
                Yom{i,1} = [Yom{i};Yo{i,j+1}];
            end
            G{i,k} = (Po{i}'*Po{i})^-1*Po{i}'*Yom{i};
            mud_bru = 1;
            % for j = 1:nf(i)
            %     for jj = 1:nf(i)
            %         if abs(A_m{i,k}(j,jj)-A_m{i,k-1}(j,jj)) > fat_alt*abs(A_m{i,k-1}(j,jj))
            %             mud_bru = 1;
            %         end
            %     end
            % end
            % for j = 1:nf(i)
            %     for jj = 1:nent
            %         if abs(B_m{i,k}(j,jj)-B_m{i,k-1}(j,jj)) > fat_alt*abs(B_m{i,k-1}(j,jj))
            %             mud_bru = 1;
            %         end
            %     end
            % end
            if mud_bru
                acionamentos = acionamentos + 1;
                xz{i,1}(:,k-tau_a) = zeros(nf(i),1);
                yet(k-tau_a,:) = (C_m{i,k}*xz{i}(:,k-tau_a)+D_m{i,k}*u(k-tau_a,:)')';
                for j = (k-tau_a+1):k
                    xz{i,1}(:,j) = A_m{i,k}*xz{i}(:,j-1)+B_m{i,k}*u(j-1,:)'-G{i,k}*(y(j-1,:)'-yet(j-1,:)');
                    yet(j,:) = (C_m{i,k}*xz{i}(:,j)+D_m{i,k}*u(j,:)')';
                end
            else
                xz{i,1}(:,k) = A_m{i,k}*xz{i}(:,k-1)+B_m{i,k}*u(k-1,:)'-G{i,k}*(y(k-1,:)'-ye(k-1,:)');
            end
            yer{i,1}(k,:) = (C_m{i,k}*xz{i}(:,k)+D_m{i,k}*u(k,:)')';
            ye(k,:) = ye(k,:) + mu(i,k)*yer{i}(k,:);
        end
    end
end
toc % Finaliza contagem de tempo
for i = 1:length(x(:,1))
    mx = max(x(i,:));
    mn = min(x(i,:));
    if mx == mn
        break
    else
        xc{1}(i,:) = xcn{end}(i,:)*(mx-mn)+mn;
    end
end

for i = 1:n
    centros_processo(i,:) = centros_processo(i,:)*(max(x(i,:))-min(x(i,:))) + min(x(i,:));
end
%% Cores para plotagens com muitas curvas
colors = [
    1.0, 0.0, 0.0;    % Vermelho
    0.0, 1.0, 0.0;    % Verde
    0.0, 0.0, 1.0;    % Azul
    1.0, 1.0, 0.0;    % Amarelo
    1.0, 0.0, 1.0;    % Magenta
    0.0, 1.0, 1.0;    % Ciano
    0.5, 0.5, 0.0;    % Oliva
    0.5, 0.0, 0.5;    % Roxo
    0.0, 0.5, 0.5;    % Verde-azulado
    0.5, 0.5, 0.5;    % Cinza
    1.0, 0.5, 0.0;    % Laranja
    0.0, 0.5, 1.0;    % Azul claro
    0.5, 0.0, 0.0;    % Marrom
    0.0, 0.5, 0.0;    % Verde escuro
    0.0, 0.0, 0.5;    % Azul escuro
    1.0, 0.75, 0.8;   % Rosa claro
    0.5, 0.75, 1.0;   % Azul céu
    0.75, 1.0, 0.75;  % Verde claro
    1.0, 0.5, 0.5;    % Coral
    0.75, 0.75, 0.0;  % Dourado
    0.75, 0.5, 0.25;  % Bronze
    0.25, 0.25, 0.75; % Índigo
    0.5, 0.25, 0.75;  % Lilás
    0.25, 0.75, 0.5;  % Verde menta
    0.75, 0.25, 0.5;  % Carmim
    0.25, 0.25, 0.25; % Preto
    0.25, 0.75, 0.75; % Turquesa
    0.75, 0.25, 0.75; % Fúcsia
    0.5, 0.75, 0.25;  % Pistache
    0.75, 0.75, 0.75; % Cinza claro
    0.75, 0.25, 0.25; % Vermelho terroso
    0.25, 0.5, 0.75;  % Azul oceano
    0.25, 0.75, 0.25; % Verde limão
    0.75, 0.5, 0.5;   % Salmão
    0.25, 0.25, 0.5;  % Azul meia-noite
    0.25, 0.5, 0.25;  % Verde militar
    0.5, 0.25, 0.25;  % Marrom avermelhado
    0.25, 0.75, 1.0;  % Azul piscina
    1.0, 0.5, 0.75;   % Rosa choque
    0.75, 1.0, 0.5;   % Verde maçã
    1.0, 1.0, 0.5;    % Amarelo pastel
    1.0, 0.75, 0.25;  % Dourado claro
    0.5, 0.25, 1.0;   % Roxo vibrante
    0.25, 1.0, 0.5;   % Verde primavera
];

%% Figura 1 (a) - Centros e Protótipos
figure
plot(u(:,1),y(:,1),'c*',xc{end}(1,:),xc{end}(3,:),'r*','LineWidth',1)
hold on
legend('Vectors of Data','Centers of Clusters','Interpreter','latex')
xlabel('Pitch Angle ($^{\circ}$)','Interpreter','latex')
ylabel('Distance (km)','Interpreter','latex')
grid on

% %% Figura 1 (b) - Dados Associados
% figure
% for i = 1:R(end)
%     plot(DA{i}(1,:),DA{i}(2,:),'.','Color',colors(i,:),'LineWidth',1);
%     if i == 1
%         hold on
%     end
%     leg{i,1} = strcat("Grupo $\mathbf{\Theta}_k^{",num2str(i),"}$");
% end
% plot(xc{end}(1,:),xc{end}(2,:),'k*','LineWidth',1)
% leg{R(end)+1,1} = "Centros de Grupo $\mathbf{c}_k^i$";
% legend(leg,'Interpreter','latex')
% ylim([0 12e5])
% yticks(0:1e5:12e5)
% xlabel('$x_k^1$','Interpreter','latex')
% ylabel('$x_k^2$','Interpreter','latex')
% grid on
% clear leg
%% Figura 1 (c) - Deslocamento de centros por processo de atualização
% figure
% plot(x(1,:),x(2,:),'.','Color',[0.8 0.7 1],'LineWidth',1)
% hold on
% plot(u_,y_,'k*','LineWidth',1)
% for i = 1:R(end)
%     plot(deslocamento{i}(1,2:end),deslocamento{i}(2,2:end),'b*',...
%         deslocamento{i}(1,1),deslocamento{i}(2,1),'g*',...
%         xc{end}(1,i),xc{end}(2,i),'r*','LineWidth',1);
% end
% plot(u_,y_,'k*','LineWidth',1)
% for i = 1:R(end)
%     plot(xc{end}(1,i),xc{end}(2,i),'r*','LineWidth',1);
% end
% legend("Dados $\mathbf{x}_k$","Centros de Grupo Reais","Centros $\mathbf{c}_k^i$ Intermedi\'{a}rios"...
%     ,"Centros $\mathbf{c}_k^i$ Iniciais","Centros $\mathbf{c}_k^i$ Finais",'Interpreter','latex')
% xlabel('$x_k^1$','Interpreter','latex')
% ylabel('$x_k^2$','Interpreter','latex')
% grid on

% figure
% plot(x(end-1,:),x(end,:),'c*','LineWidth',1)
% hold on
% plot(centros_processo(end-1,:),centros_processo(end,:),'b*','LineWidth',1)
% plot(u_,y_,'ko','LineWidth',1)
% plot(xc{end}(end-1,:),xc{end}(end,:),'r*','LineWidth',1)
% legend("Vectors of Data","Centers in the Process",...
%     "Real Centers","Final Centers",'Interpreter','latex')
% xlabel('$x_k^1$','Interpreter','latex')
% ylabel('$x_k^2$','Interpreter','latex')
% grid on
% yticks_ = get(gca, 'YTick'); % Obter valores dos ticks no eixo Y
% yticklabels = strrep(strtrim(cellstr(num2str(yticks_.', '%.2f'))), '.', ',');
% set(gca, 'YTickLabel', yticklabels);
% xticks_ = get(gca, 'XTick'); % Obter valores dos ticks no eixo Y
% xticklabels = strrep(strtrim(cellstr(num2str(xticks_.', '%.2f'))), '.', ',');
% set(gca, 'XTickLabel', xticklabels);

% %% Figura 1 (d) - Niveis de Contorno
% % Gera curva de pertinencias
% figure
% plot(x(1,:),x(2,:),'c*','LineWidth',1)
% hold on
% plot(xc{end}(1,:),xc{end}(2,:),'r*','LineWidth',1)
% xm = (-0.1:0.005:1.1)';
% xmnn = xm*(max(x(1,:))-min(x(1,:))) + min(x(1,:));
% ym = (-0.1:0.005:1.1)';
% ymnn = ym*(max(x(2,:))-min(x(2,:))) + min(x(2,:));
% for i = 1:length(xm)
%     i
%     for j = 1:length(ym)
%         for l = 1:R(end)
%             Dm(i,j,l) = exp(sqrt(([xm(i);ym(j)]-xcn{end}(:,l))'*det(F{end}(:,:,l))^(1/n)*inv(F{end}(:,:,l))*...
%                 ([xm(i);ym(j)]-xcn{end}(:,l))))-1;
%         end
%         % Calcula a pertinencia do dado k em relação a todos os clusters
%         indcn = 0; % Inicializa o indice de cluster com distancia nula
%         for l = 1:R(end) % Para todos os clusters
%             if Dm(i,j,l) == 0 % Se a distancia for nula
%                 indcn = l; % Detecta cluster com distancia nula
%             end
%         end
%         if indcn == 0 % Se nao houver centro de cluster igual ao dado atual
%             for l = 1:R(end) % Para todos os clusters
%                 if ~isinf(Dm(i,j,l)) % Se a distancia nao for infinita
%                     mum(i,j,l) = 0; % Inicializa pertinencia
%                     for ll = 1:R(end) % Para todos os clusters
%                         mum(i,j,l) = mum(i,j,l) + (Dm(i,j,l)/Dm(i,j,ll))^(2/(m-1)); % Calcula o inverso da pertinencia
%                     end
%                     mum(i,j,l) = 1/mum(i,j,l); % Calcula a pertinencia
%                 else % Se a distancia for infinita
%                     mum(i,j,l) = 0; % Define pertinencia nula
%                 end
%             end
%         else % Se houver centro de cluster igual ao dado atual
%             mum(i,j,indcn) = 1; % Define pertinencia unitaria para este cluster
%         end
%     end
% end
% for l = 1:R(end)
%     contour(xmnn,ymnn,mum(:,:,l)',0.9*[1,1],'r','LineWidth',1);
%     contour(xmnn,ymnn,mum(:,:,l)',0.7*[1,1],'b','LineWidth',1);
%     contour(xmnn,ymnn,mum(:,:,l)',0.5*[1,1],'k','LineWidth',1);
% end
% legend("Vectors of Data","Centers of Clusters",...
%     "$\mu$ = 0.9","$\mu$ = 0.7"...
%     ,"$\mu$ = 0.5",'Interpreter','latex')
% xlabel('$x_k^1$','Interpreter','latex')
% ylabel('$x_k^2$','Interpreter','latex')
% grid on

% %% Figura 2 (a) - Dados em série temporal
% figure
% subplot(2,1,1)
% plot(t,u/1e5,'k','LineWidth',1)
% xlabel('$k$','Interpreter','latex')
% ylabel('$x_k^1$','Interpreter','latex')
% % axis([0 4130 0 5])
% yticks(0:2.5:10)
% grid on
% yticks_ = get(gca, 'YTick'); % Obter valores dos ticks no eixo Y
% yticklabels = strrep(strtrim(cellstr(num2str(yticks_.', '%.2f'))), '.', ',');
% set(gca, 'YTickLabel', yticklabels);
% % yticks(0:1:5)
% subplot(2,1,2)
% plot(t,y/1e5,'k','LineWidth',1)
% xlabel('$k$','Interpreter','latex')
% ylabel('$x_k^2$','Interpreter','latex')
% % axis([0 4130 -2 2])
% yticks(0:2.5:10)
% grid on
% yticks_ = get(gca, 'YTick'); % Obter valores dos ticks no eixo Y
% yticklabels = strrep(strtrim(cellstr(num2str(yticks_.', '%.2f'))), '.', ',');
% set(gca, 'YTickLabel', yticklabels);

%% Figura 2 (b) - Variação do Número de Grupos
figure
t = (1:N)';
plot(t,R,'b','LineWidth',1)
xlabel('$k$','Interpreter','latex')
ylabel('$R$','Interpreter','latex')
ylim([0 5])
yticks(0:1:5)
xlim([0 4358])
xticks(0:400:4000)
grid on

% %% Figura 2 (c) - Potenciais dos dados
% figure
% plot(t(2:end),P(2:end),'k','LineWidth',1)
% xlabel('$k$','Interpreter','latex')
% ylabel('$P_k^{\bar{\mathbf{x}}_k}$','Interpreter','latex')
% % ylim([0 17])
% % yticks(0:1:17)
% grid on
% yticks_ = get(gca, 'YTick'); % Obter valores dos ticks no eixo Y
% yticklabels = strrep(strtrim(cellstr(num2str(yticks_.', '%.2f'))), '.', ',');
% set(gca, 'YTickLabel', yticklabels);

% %% Figura 2 (d) - Potenciais dos centros no proceranksso
% figure
% PotC = zeros(max(R),N);
% for k = 1:N
%     if length(Pc{k}) == max(R)
%         PotC(:,k) = Pc{k};
%     elseif length(Pc{k}) < max(R)
%         PotC(:,k) = [Pc{k};zeros(max(R)-length(Pc{k}),1)];
%     else
%         PotC(:,k) = Pc{k}(1:max(R));
%     end
% end
% for i = 1:max(R)
%     plot(t,PotC(i,:),'Color',colors(i,:),'LineWidth',1);
%     if i == 1
%         hold on
%     end
%     leg{i,1} = strcat("Grupo $\mathbf{\Theta}_k^{",num2str(i),"}$");
% end
% legend(leg,'Interpreter','latex')
% xlabel('$k$','Interpreter','latex')
% ylabel('$P_k^{\bar{\mathbf{c}}_k^i}$','Interpreter','latex')
% grid on
% yticks_ = get(gca, 'YTick'); % Obter valores dos ticks no eixo Y
% yticklabels = strrep(strtrim(cellstr(num2str(yticks_.', '%.2f'))), '.', ',');
% set(gca, 'YTickLabel', yticklabels);

% %% Figura 3 (a) - Variação dos Graus de Pertinência
% 
% figure
% for i = 1:max(R)
%         plot(t,muf(i,:),'Color',colors(i,:),'LineWidth',1);
%     if i == 1
%         hold on
%     end
% leg{i,1} = strcat("Grupo $\mathbf{\Theta}_k^{",num2str(i),"}$");
% end
% legend(leg,'Interpreter','latex')
% xlabel('$k$','Interpreter','latex')
% ylabel('$\mu_k^i$','Interpreter','latex')
% ylim([0 1.2]);
% yticks(0:0.1:1.2);
% grid on
% yticks_ = get(gca, 'YTick'); % Obter valores dos ticks no eixo Y
% yticklabels = strrep(strtrim(cellstr(num2str(yticks_.', '%.2f'))), '.', ',');
% set(gca, 'YTickLabel', yticklabels);

% %% Figura 3 (b) - Suportes no processo
% figure
% SC = zeros(max(R),N);
% for k = 1:N
%     if length(S{k}) == max(R)
%         SC(:,k) = S{k};
%     elseif length(S{k}) < max(R)
%         SC(:,k) = [S{k};zeros(max(R)-length(S{k}),1)];
%     else
%         SC(:,k) = S{k}(1:max(R));
%     end
% end
% for i = 1:max(R)
%     plot(t,SC(i,:),'Color',colors(i,:),'LineWidth',1);
%     if i == 1
%         hold on
%     end
%     leg{i,1} = strcat("Grupo $\mathbf{\Theta}_k^{",num2str(i),"}$");
% end
% legend(leg,'Interpreter','latex')
% xlabel('$k$','Interpreter','latex')
% ylabel('$S_k^i$','Interpreter','latex')
% grid on

% %% Figura 3 (c) - Raios dos grupos no processo
% figure
% epsilonC = zeros(max(R),N);
% for k = 1:N
%     if length(epsilon{k}) == max(R)
%         epsilonC(:,k) = epsilon{k};
%     elseif length(Pc{k}) < max(R)
%         epsilonC(:,k) = [epsilon{k};zeros(max(R)-length(epsilon{k}),1)];
%     else
%         epsilonC(:,k) = epsilon{k}(1:max(R));
%     end
% end
% for i = 1:max(R)
%     plot(t,epsilonC(i,:),'Color',colors(i,:),'LineWidth',1);
%     if i == 1
%         hold on
%     end
%     leg{i,1} = strcat("Grupo $\mathbf{\Theta}_k^{",num2str(i),"}$");
% end
% legend(leg,'Interpreter','latex')
% xlabel('$k$','Interpreter','latex')
% ylabel('$\varepsilon_k^i$','Interpreter','latex')
% grid on
% yticks_ = get(gca, 'YTick'); % Obter valores dos ticks no eixo Y
% yticklabels = strrep(strtrim(cellstr(num2str(yticks_.', '%.2f'))), '.', ',');
% set(gca, 'YTickLabel', yticklabels);

% %% Figura 3 (d) - Sensibilidades no processo
% figure
% sC = zeros(max(R),N);
% for k = 1:N
%     if length(s{k}) == max(R)
%         sC(:,k) = s{k};
%     elseif length(s{k}) < max(R)
%         sC(:,k) = [s{k};zeros(max(R)-length(s{k}),1)];
%     else
%         sC(:,k) = s{k}(1:max(R));
%     end
% end
% for i = 1:max(R)
%     plot(t,sC(i,:),'Color',colors(i,:),'LineWidth',1);
%     if i == 1
%         hold on
%     end
%     leg{i,1} = strcat("Cluster $",num2str(i),"$");
% end
% legend(leg,'Interpreter','latex')
% xlabel('$k$','Interpreter','latex')
% ylabel('$\sigma_i$','Interpreter','latex')
% ylim([0 0.5]);
% yticks(0:0.05:0.5);
% grid on

%% Figura 4 (a) - Estimação da saída
figure
    plot(t,y(:,1),'k',t,ye(:,1),'b--','LineWidth',1)
xlabel('$k$','Interpreter','latex')
ylabel('Distance (km)','Interpreter','latex')
legend('$y_k$','$\hat{y}_k$','Interpreter','latex')
grid on
xlim([0 4358])
xticks(0:400:4000)

% %% Figura 4 (b) - Erro
% erro_medio(Ni+1,1) = (y(Ni+1)-ye(Ni+1))^2;
% raiz_erro_medio(Ni+1,1) = sqrt(erro_medio(Ni+1,1));
% j = 1;
% for i = Ni+2:N
%     j = j + 1;
%     erro_medio(i,1) =  ((j-1)*erro_medio(i-1,1) + (y(i)-ye(i))^2)/j;
%     raiz_erro_medio(i,1) = sqrt(erro_medio(i,1));
% end
% figure
% plot(t(Ni+1:end),y(Ni+1:end)-ye(Ni+1:end),'k',t(Ni+1:end),raiz_erro_medio(Ni+1:end),'g','LineWidth',1)
% legend('$e_k$','RMSE','Interpreter','latex')
% xlabel('$k$','Interpreter','latex')
% ylabel('$e_k$','Interpreter','latex')
% xlim([Ni 2900])
% xticks(Ni:300:2700)
% yticks_ = get(gca, 'YTick'); % Obter valores dos ticks no eixo Y
% yticklabels = strrep(strtrim(cellstr(num2str(yticks_.', '%.2f'))), '.', ',');
% set(gca, 'YTickLabel', yticklabels);
% grid on

figure
subplot(2,1,1)
plot(t,u(:,1),'b','LineWidth',1)
legend('$u_k^1$','Interpreter','latex')
xlabel('$k$','Interpreter','latex')
ylabel('Pitch Angle ($^{\circ}$)','Interpreter','latex')
grid on
xlim([0 4358])
xticks(0:400:4000)
subplot(2,1,2)
plot(t,u(:,2),'b','LineWidth',1)
legend('$u_k^2$','Interpreter','latex')
xlabel('$k$','Interpreter','latex')
ylabel('Yaw Angle ($^{\circ}$)','Interpreter','latex')
grid on
xlim([0 4358])
xticks(0:400:4000)


%% Computa o índice de qualidade e de proximidade de centros
% Mede a partição final
mu = zeros(R(end),N);
D_ = mu;
for i = 1:R(end)
    dadosg{i,1} = [];
end
for j = 1:N
    for i = 1:R(end)  % Para todos os centros de clusters
        dist = sqrt((xn(:,j)-xcn{end}(:,i))'*det(F{end}(:,:,i))^(1/n)*inv(F{end}(:,:,i))*...
            (xn(:,j)-xcn{end}(:,i))); % Norma induzida
        D_(i,j) = exp(dist)-1; % Medicao da distancia adaptativa do dado atual para os centros
    end
    indcn = 0; % Inicializa o indice de cluster com distancia nula
    for i = 1:R(end) % Para todos os clusters
        if D_(i,j) == 0 % Se a distancia for nula
            indcn = i; % Detecta cluster com distancia nula
        end
    end
    if indcn == 0 % Se nao houver centro de cluster igual ao dado atual
        for i = 1:R(end) % Para todos os clusters
            if ~isinf(D_(i,j)) % Se a distancia nao for infinita
                mu(i,j) = 0; % Inicializa pertinencia
                for jj = 1:R(end) % Para todos os clusters
                    mu(i,j) = mu(i,j) + (D_(i,j)/D_(jj,j))^(2/(m-1)); % Calcula o inverso da pertinencia
                end
                mu(i,j) = 1/mu(i,j); % Calcula a pertinencia
            else % Se a distancia for infinita
                mu(i,j) = 0; % Define pertinencia nula
            end
        end
    else % Se houver centro de cluster igual ao dado atual
        mu(indcn,j) = 1; % Define pertinencia unitaria para este cluster
    end
    indcmaxmu = 0;
    mumax = 0;
    for i = 1:R(end)
        if mu(i,j) > mumax
            mumax = mu(i,j);
            indcmaxmu = i;
        end
    end
    dadosg{indcmaxmu} = [dadosg{indcmaxmu},x(:,j)];
end

% figure
% for i = 1:R(end)
%     plot(dadosg{i}(1,:),dadosg{i}(2,:),'.','Color',colors(i,:),'LineWidth',1);
%     if i == 1
%         hold on
%     end
%     leg{i,1} = strcat("Grupo $\mathbf{\Theta}_k^{",num2str(i),"}$");
% end
% plot(xc{end}(1,:),xc{end}(2,:),'k*','LineWidth',1)
% leg{R(end)+1,1} = "Centros de Grupo $\mathbf{c}_k^i$";
% legend(leg,'Interpreter','latex')
% ylim([0 12e5])
% yticks(0:1e5:12e5)
% xlabel('$x_k^1$','Interpreter','latex')
% ylabel('$x_k^2$','Interpreter','latex')
% grid on
% clear leg

dmin = 1e100;
clear Dc
Dc = zeros(R(end),R(end)); % Inicializa as ditâncias
for i = 1:R(end)
    for j = 1:R(end) % Calcula as distâncias entre centros medidas a partir de cada um
        Dc(i,j) = norm(xc{end}(:,i)-xc{end}(:,j));
        if (i ~= j) && (Dc(i,j)<dmin) % Encontra a menor das distancias
            i_cp = [i,j];
            dmin = Dc(i,j);
        end
    end
end
num = 0;
for i = 1:R(end)
    for j = 1:N
        num = num + mu(i,j)^2*norm(xc{end}(:,i)-x(:,j));
    end
end
q_c = num/(N*dmin)


% RMSE1 = sqrt(mean((y(Ni+1:end,1)-ye(Ni+1:end,1)).^2));
% NDEI1 = RMSE1/std(ye(Ni+1:end,1))
% RMSE2 = sqrt(mean((y(Ni+1:end,2)-ye(Ni+1:end,2)).^2));
% NDEI2 = RMSE1/std(ye(Ni+1:end,2))
VAF1 = 1 - var(y(Ni+1:end,1)-ye(Ni+1:end,1))/var(y(Ni+1:end,1))
% VAF2 = 1 - var(y(Ni+1:end,2)-ye(Ni+1:end,2))/var(y(Ni+1:end,2))


e = y-ye;

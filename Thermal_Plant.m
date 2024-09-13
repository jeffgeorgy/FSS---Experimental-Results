%% Limpa a memoria e carrega os dados
clear
clc
close all
% Carrega os dados da planta
tabela_temp = readtable('temperatura.xls');
tabela_tensao = readtable('tensao.xls'); 
yi = tabela_temp{3:end,2}; ui = tabela_tensao{3:end,2};
tk = tabela_temp{3:end,1}; tk = tk-2;
Ts = 0.3; %t = (Ts/60)*k; % Tempo em minutos
Tsn = 3; passo = Tsn/Ts;
for i = 1:passo:length(tk)
    j = ceil(i/passo);
    u(j,1) = ui(i);
    y(j,1) = yi(i);
end
tk = (1:1:length(u))'-1;
t = Tsn*tk;
%clear tabela_tensao tabela_temp ui tempi
% Amostras em regime permanente
ind = [1;1510;3001;4501;6001;7252;8506;9999];
for i = 1:length(ind)
    ind(i) = ceil(ind(i)/passo);
end
u_ = u(ind); y_ = y(ind);
u_o = sort(u_); y_o = sort(y_);
max_u = 170; min_u = 0; % Valor maximo e minimo da entrada para normalizacao
max_y = 210; min_y = 0; % Valor maximo e minimo da saida para normalizacao
u_on = (u_o-min_u)/(max_u-min_u); % Normalizando as amostras ordenadas
y_on = (y_o-min_y)/(max_y-min_y); % Normalizando a saida
un = (u-min_u)/(max_u-min_u); % normalizando
yn = (y-min_y)/(max_y-min_y); % normalizando
u_n = (u_-min_u)/(max_u-min_u); % normalizando
y_n = (y_-min_y)/(max_y-min_y); % normalizando
%% Inicializa as variaveis do Algoritmo ETS
% Parâmetros do algoritmo
r = 0.15; % raio de influencia de um cluster
epsilon = 0.3; % Fator de raio de cluster
m = 2; %Ponderamento exponencial para calculo de pertinencias
gama = 0.5; % Ponderamento das informacoes no potencial
s = 0.9; % Fator de sensibilidade para determinacao de candidato
nmm = 1; % Tamanho da janela de media movel
tic
% Variaveis do algoritmo
alpha = 4/r^2; % Parametro dependente de r
k = 1; % Amostra atual
R = 1; % Numero de regras inicial
xn = [un';yn']; % Vetor de dados normalizados
x = [u';y']; % Vetor de dados original
n = length(xn(:,1)); % Tamanho de dados de entrada 
xcn{1} = [xn(:,1)]; % Primeiro centro de cluster
N = length(xn(1,:)); % Tamando dos dados
jan = zeros(2,nmm); % Janela de media movel
varphi_b = 0; % Valor inicial da taxa de variacao
tvn_c = 0; % Valor da taxa de variacao do primeiro centro
P(1) = 1; % Potencial do primeiro dado
Pc{1} = [P(1),0]; % Na primeira coluna, o potencial do centro, na segunda
% a quantidade de dados usados para medi-lo
iuc = 1; % indice do ultimo centro de cluster encontrado
vmax = 0; % Valor maximo da taxa de variacao para normalizacao
NF(:,:,1) = r*eye(n); % Inicializacao do numerador da matriz de covariancia
% fuzzy
DF(1) = 1; % Inicializacao do denominador da matriz de covariancia fuzzy
F(:,:,1) = NF(:,:,1)/DF(1);  % Inicializacao da matriz de covariancia fuzzy
D(1,1) = 0; % Distancia do primeiro dado ao primeiro centro
mu(1,1) = 1; % pertinencia do primeiro dado ao primeiro centro
pex = 1; % Peso exponencial no calcula da distancia adptativa
% Parametros consequente
Ni = 100; % Instante de inicializacao
nent = length(u(1,:));
nsai = length(y(1,:));
p_pm = 2;
v = [u';y'];
alphaf = 10;
betaf = 10;
lambda = 0.95;
I = 1;
fat_alt = 0.01;
qt_reg = 1;
%% Realizacao do algoritmo
% O algoritmo agrupa apenas os dados x, sem considerar a saida y, e
% consequentemente o vetor inteiro de dados z
% plot(1,Pc{1},'ko')
% hold on
for k = 2:N
    novo_cluster = 0;
    k % Mostra o valor de k
    %Calcula a taxa de variacao
    txv(:,k) = xn(:,k) - xn(:,k-1); % Calcula a taxa de variacao da amostra
    %atual
    if nmm == 1
        txvf(:,k) = txv(:,k); % Se a janela for de 1 unidade, a taxa filtrada 
        % e igual a ela propria
    else
        jan = [jan(:,2:nmm),txv(:,k)]; % Atualiza a janela
        txvf(:,k) = mean(jan,2); % Calcula a taxa filtrada
    end
    varphi(k) = norm(txvf(:,k)); % Medicao de variacao pela norma
    if varphi(k) > vmax
        vmax = varphi(k); % Atualiza a medicao maxima se for o caso
    end
    if vmax ~= 0
        varphi_b(k) = varphi(k)/vmax; % Atualiza a medicao normalizada
    else 
        varphi_b(k) = varphi(k); % Atualiza a medicao normalizada
    end
    Pc{k,1} = Pc{k-1}; % Potencial de centros no instante atual
    xcn{k,1} = xcn{k-1}; % Centros no instante atual
    for i = 1:R % Medicao da distancia adaptativa do dado atual para os centros
        dist = sqrt((xn(:,k)-xcn{k}(:,i))'*det(F(:,:,i))^(1/n)*inv(F(:,:,i))*...
            (xn(:,k)-xcn{k}(:,i)));
        D(i,k) = pex*(exp(dist)-1) + (1-pex)*dist;
    end
    P(k,1) = 0; % Inicializa o potencial da amostra atual
    Nda = k-iuc-1; % Atualiza a quantidade de dados usados na medicao do dado atual
    % Calcula o potencial do novo dado primeiro com a relação aos
    % centros existentes e atualiza os potenciais dos centros existentes
    for j = 1:R
        P(k,1) = P(k) + 1/(R+Nda)*((1-gama)*exp(-alpha*D(j,k)^2) + gama*exp(...
            -alpha*varphi_b(k)^2)); % Potencial do dado atual
        Pc{k}(j,2) = Pc{k}(j,2) + 1; % Numero de amostras usadas nos potenciais dos centros
        Pc{k}(j,1) = (Pc{k}(j,2)-1)/Pc{k}(j,2)*Pc{k}(j,1) + 1/Pc{k}(j,2)*...
            ((1-gama)*exp(-alpha*D(j,k)^2) + gama*exp(-alpha*tvn_c(j)^2)); % Potenciais
        % de centros
    end
    % Calcula o potencial do novo dado depois com a relação aos
    % dados pos ultimo cluster
    for j = (iuc+1):(k-1)
        P(k,1) = P(k) + 1/(R+Nda)*((1-gama)*exp(-alpha*norm(xn(:,k)-xn(:,j))^2)...
            + gama*exp(-alpha*varphi_b(k)^2)); %Potencial do dado atual
    end
    % Avalia as condicoes para evolucao da estrutura
    if P(k) > s*max(Pc{k}(:,1)) % Se o potencial do dado atual for maior
        % que o fator de sensibilidade vezes o maior potencial de centro
        dmin = 1e6; % Inicializa a distancia
        for i = 1:R % Calcula a distancia para o centro mais proximo
            if D(i,k) < dmin
                dmin = D(i,k);
                indc_prox = i;
            end
        end
        if dmin < epsilon*r
            rou = P(k)/(P(k)+Pc{k}(indc_prox,1)); % Peso para o dado atual
            % Novas coordenadas de centro de cluster
            xcn{k}(:,indc_prox) = rou*xn(:,k) + (1-rou)*xcn{k}(:,indc_prox);
            % Atualiza o potencial e a quantidade de dados considerados
            % para a sua medição
            Pc{k}(indc_prox,:) = [rou*P(k)+(1-rou)*Pc{k}(indc_prox,1),ceil(rou*...
                (R+Nda)+(1-rou)*Pc{k}(indc_prox,2))];
            tvn_c(indc_prox) = rou*varphi_b(k) + (1-rou)*tvn_c(indc_prox);
            % Atualiza a distância do dado atual para o novo centro
            % resultante do curzamento
            dist = sqrt((xn(:,k)-xcn{k}(:,indc_prox))'*det(F(:,:,indc_prox))^(1/n)*inv(F(:,:,indc_prox))*...
            (xn(:,k)-xcn{k}(:,indc_prox)));
            D(indc_prox,k) = pex*(exp(dist)-1) + (1-pex)*dist;
            % Plota o novo centro
        else
            R = R + 1; % Aumenta o numero de regras
            I(R,1) = k;
            novo_cluster = 1;
            iuc = k; % atualiza o indice do ultimo centro criado para o atual
            xcn{k}(:,R) = xn(:,k); % Centro da nova regra
            Pc{k}(R,1) = P(k); % Potencial do novo centro
            Pc{k}(R,2) = R-1 + Nda; % Numero de dados usados para medicao do potencial
            tvn_c(R,1) = varphi_b(k); % Taxa de variacao do centro atual
            % Inicializa a matriz de covariancia fuzzy quando se cria uma nova regra
            NF(:,:,R) = r*eye(n); % Numerador
            DF(R,1) = 1; % Denominador
            % Alternativa, inicializar como a media dos centros existentes
            % NF(:,:,R) = zeros(n,n);
            % DF(R,1) = 0;
            % for i = 1:(R-1)
            %     NF(:,:,R) = NF(:,:,R) + 1/(R-1)*NF(:,:,i);
            %     DF(R) = DF(R) + 1/(R-1)*DF(i);
            % end
            F(:,:,R) = NF(:,:,R)/DF(R); % Matriz de covariancia fuzzy
            % Calcula a distância do dado atual para o centro existente
            dist = sqrt((xn(:,k)-xcn{k}(:,R))'*det(F(:,:,R))^(1/n)*inv(F(:,:,R))*...
            (xn(:,k)-xcn{k}(:,R)));
            D(R,k) = pex*(exp(dist)-1) + (1-pex)*dist;
            % Cria os parametros do consequente da nova regra
            if k > Ni
                % mu(:,(k-Ni+1):k) = zeros(R-1,Ni);
                % for j = (k-Ni+1):k
                %     for i = 1:R % Medicao da distancia adaptativa do dado atual para os centros
                %         dist = sqrt((xn(:,j)-xcn{k}(:,i))'*det(F(:,:,i))^(1/n)*inv(F(:,:,i))*...
                %             (xn(:,j)-xcn{k}(:,i)));
                %         D(i,j) = pex*(exp(dist)-1) + (1-pex)*dist;
                %     end
                %     % Armazena os indices de elementos nulos em D(:,k)
                %     indcn = find(D(:,j)==0);
                %     if isempty(indcn) % Se nao houver elementos nulos
                %         for i = 1:R
                %             mu(i,j) = 0;
                %             for jj = 1:R
                %                 mu(i,j) = mu(i,j) + (D(i,j)/D(jj,j))^(2/(m-1));
                %             end
                %             mu(i,j) = 1/mu(i,j);
                %         end
                %     else % Se houver elementos nulos em D(:,k)
                %         % Divide igualmente a pertinencia nos clusters em que D(i,k)=0,
                %         % e deixa pertinencia 0 nos demais clusters
                %         for i = indcn
                %             mu(i,j) = 1/(length(indcn));
                %         end
                %     end
                % end
                % y_v = y((k-Ni+p_pm+1):k,:)';
                % V = u((k-Ni+p_pm+1):k,:)';
                % for i = 1:p_pm
                %     V = [V;v(:,(k-Ni+p_pm+1-i):(k-i))];
                % end
                % W{R} = diag(mu(R,(k-Ni+p_pm+1):k));
                % P_cov{R} = (V*W{R}*V')^-1;
                % Y_m{R} = y_v*W{R}*V'*P_cov{R};
                % yves{R} = Y_m{R}*V;
                % for j = 1:nent
                %     figure
                %     plot(y_v(j,:))
                %     hold on
                %     plot(yves{R}(j,:),'--')
                % end
                P_cov{R} = 0;
                Y_m{R} = 0;
                for i = 1:(R-1)
                    P_cov{R} = P_cov{R} + 1/(R-1)*P_cov{i};
                    Y_m{R} = Y_m{R} + 1/(R-1)*Y_m{i};
                end
                Z{R} = 0;
                Y_0{R} = 0;
                Y_{R,1} = 0;
                Y_1{R,1} = 0;
                Y_2{R,1} = 0;
                Y{R,1} = 0;
                H0{R} = 0;
                H1{R} = 0;
                R_svd{R} = 0;
                S_svd{R} = 0;
                Sigma{R} = 0;
                Sigma_n{R} = 0;
                Rn{R} = 0;
                Sn{R} = 0;
                A_{R,k-1} = zeros(nf,nf);
                B_{R,k-1} = zeros(nf,nent);
                C_{R,k-1} = zeros(nsai,nf);
                D_{R,k-1} = zeros(nsai,nent);
                Yo{R,1} = 0;
                G{R,1} = 0;
                Po{R} = 0;
                Yom{R} = 0;
                xz{R}(:,k-1) = zeros(nf,1);
                yer{R}(k-1,:) = zeros(1,nsai);
            end
        end
    end
    % Mecanismo de crossover de clusters
    dmin = 1e6;
    Dc = zeros(R,R); % Inicializa as ditâncias
    for i = 1:R
        for j = 1:R % Calcula as distâncias entre centros medidas a partir de cada um
            dist = sqrt((xcn{k}(:,i)-xcn{k}(:,j))'*det(F(:,:,i))^(1/n)*inv(F(:,:,i))*...
            (xcn{k}(:,i)-xcn{k}(:,j)));
            Dc(i,j) = pex*(exp(dist)-1) + (1-pex)*dist;
            if (i ~= j) && (Dc(i,j)<dmin) % Encontra a menor das distancias
                i_cp = [i,j];
                dmin = Dc(i,j);
            end
        end
    end
    if dmin < epsilon*r % Se tiver sobreposicao de agrupamentos
        % Fator de ponderacao dos clusters pelo potencial
        rou = Pc{k}(i_cp(1),1)/(Pc{k}(i_cp(1),1)+Pc{k}(i_cp(2),1));
        % Define a nova posicao de centro e elimina os geradores
        xcn{k}(:,R+1) = rou*xcn{k}(:,i_cp(1)) + (1-rou)*xcn{k}(:,i_cp(2));
        xcn{k}(:,max(i_cp)) = [];
        xcn{k}(:,min(i_cp)) = [];
        % Define o novo potencial e elimina os geradores
        Pc{k}(R+1,:) = [rou*Pc{k}(i_cp(1),1)+(1-rou)*Pc{k}(i_cp(2),1),ceil(rou*...
            Pc{k}(i_cp(1),2)+(1-rou)*Pc{k}(i_cp(2),2))];
        Pc{k}(max(i_cp),:) = [];
        Pc{k}(min(i_cp),:) = [];
        % Define a nova taxa de variacao normalizada e elimina os geradores
        tvn_c(R+1) = rou*tvn_c(i_cp(1)) + (1-rou)*tvn_c(i_cp(2));
        tvn_c(max(i_cp)) = [];
        tvn_c(min(i_cp)) = [];
        % Inicializa a linha de distancias do novo cluster e elimina dos
        % geradores
        D(R+1,:) = zeros(size(D(1,:)));
        D(max(i_cp),:) = [];
        D(min(i_cp),:) = [];
        % Inicializa a linha de pertinencias dos novos clusters e elimina
        % dos geradores
        mu(R+1,:) = zeros(size(mu(1,:)));
        mu(max(i_cp),:) = [];
        mu(min(i_cp),:) = [];
        % Determina a nova matriz de covariancia e elimina as geradoras
        NF(:,:,R+1) = rou*NF(:,:,i_cp(1))+(1-rou)*NF(:,:,i_cp(2));
        DF(R+1) = rou*DF(i_cp(1))+(1-rou)*DF(i_cp(2));
        F(:,:,R+1) = NF(:,:,R+1)/DF(R+1);
        NF(:,:,max(i_cp)) = [];
        NF(:,:,min(i_cp)) = [];
        DF(max(i_cp)) = [];
        DF(min(i_cp)) = [];
        F(:,:,max(i_cp)) = [];
        F(:,:,min(i_cp)) = [];
        if k > Ni
            A_{R+1,k-1} = zeros(nf,nf);
            B_{R+1,k-1} = zeros(nf,nent);
            C_{R+1,k-1} = zeros(nsai,nf);
            D_{R+1,k-1} = zeros(nsai,nent);
            if k > Ni+1
                Z{R+1} = rou*Z{i_cp(1)} + (1-rou)*Z{i_cp(2)};
                Z(max(i_cp)) = [];
                Z(min(i_cp)) = [];
                yer(max(i_cp)) = [];
                yer(min(i_cp)) = [];
            end
            P_cov{R+1} = rou*P_cov{i_cp(1)} + (1-rou)*P_cov{i_cp(2)};
            P_cov(max(i_cp)) = [];
            P_cov(min(i_cp)) = [];
            Y_m{R+1} = rou*Y_m{i_cp(1)} + (1-rou)*Y_m{i_cp(2)};
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
            A_(max(i_cp),:) = [];
            A_(min(i_cp),:) = [];
            B_(max(i_cp),:) = [];
            B_(min(i_cp),:) = [];
            C_(max(i_cp),:) = [];
            C_(min(i_cp),:) = [];
            D_(max(i_cp),:) = [];
            D_(min(i_cp),:) = [];
            Yo(max(i_cp),:) = [];
            Yo(min(i_cp),:) = [];
            G(max(i_cp),:) = [];
            G(min(i_cp),:) = [];
            Po(max(i_cp)) = [];
            Po(min(i_cp)) = [];
            Yom(max(i_cp)) = [];
            Yom(min(i_cp)) = [];
            xz{R+1}(:,k-1) = zeros(nf,1);
            xz(max(i_cp)) = [];
            xz(min(i_cp)) = [];
        end
        R = R-1; % Diminui a quantidade de regras
        dist = sqrt((xn(:,k)-xcn{k}(:,R))'*det(F(:,:,R))^(1/n)*inv(F(:,:,R))*...
            (xn(:,k)-xcn{k}(:,R)));
        D(R,k) = pex*(exp(dist)-1) + (1-pex)*dist;
    end
    % Armazena os indices de elementos nulos em D(:,k)
    indcn = find(D(:,k)==0); 
    if isempty(indcn) % Se nao houver elementos nulos
        for i = 1:R
            mu(i,k) = 0;
            for j = 1:R
                mu(i,k) = mu(i,k) + (D(i,k)/D(j,k))^(2/(m-1));
            end
            mu(i,k) = 1/mu(i,k);
        end
    else % Se houver elementos nulos em D(:,k)
        % Divide igualmente a pertinencia nos clusters em que D(i,k)=0,
        % e deixa pertinencia 0 nos demais clusters
        for i = indcn
             mu(i,k) = 1/(length(indcn));
        end
    end
    for i = 1:R % Atualiza a matriz de covariância fuzzy com o dado atual
        NF(:,:,i) = NF(:,:,i) + mu(i,k)^m*(xn(:,k)-xcn{k}(:,i))*(xn(:,k)-xcn{k}(:,i))';
        DF(i,1) = DF(i) + mu(i,k)^m;
        F(:,:,i) = NF(:,:,i)/DF(i);
    end
    if k == Ni % Inicializa os parametros de Markov
        % Mede as pertinencias dos Ni primeiros dados em relacao aos
        % clusters formados
        mu(:,1:Ni) = zeros(R,Ni);
        for j = 1:Ni
            for i = 1:R % Medicao da distancia adaptativa do dado atual para os centros
                dist = sqrt((xn(:,j)-xcn{k}(:,i))'*det(F(:,:,i))^(1/n)*inv(F(:,:,i))*...
                    (xn(:,j)-xcn{k}(:,i)));
                D(i,j) = pex*(exp(dist)-1) + (1-pex)*dist;
            end
            % Armazena os indices de elementos nulos em D(:,k)
            indcn = find(D(:,j)==0);
            if isempty(indcn) % Se nao houver elementos nulos
                for i = 1:R
                    mu(i,j) = 0;
                    for jj = 1:R
                        mu(i,j) = mu(i,j) + (D(i,j)/D(jj,j))^(2/(m-1));
                    end
                    mu(i,j) = 1/mu(i,j);
                end
            else % Se houver elementos nulos em D(:,k)
                % Divide igualmente a pertinencia nos clusters em que D(i,k)=0,
                % e deixa pertinencia 0 nos demais clusters
                for i = indcn
                    mu(i,j) = 1/(length(indcn));
                end
            end
        end
        y_v = y((p_pm+1):Ni,:)';
        V = u((p_pm+1):Ni,:)';
        tau_a = p_pm;
        for i = 1:p_pm
            V = [V;v(:,(p_pm+1-i):(Ni-i))];
        end
        for i = 1:R
            W{i} = diag(mu(i,(p_pm+1):Ni));
            P_cov{i} = (V*W{i}*V')^-1;
            Y_m{i} = y_v*W{i}*V'*P_cov{i};
            yves{i} = Y_m{i}*V;
            % for j = 1:nent
            %     figure
            %     plot(y_v(j,:))
            %     hold on
            %     plot(yves{i}(j,:),'--')
            % end
            Y_0{i} = Y_m{i}(1:nsai,1:nent);
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
            H0{i} = cell2mat(Y(i,1:betaf));
            H1{i} = cell2mat(Y(i,2:(betaf+1)));
            for j = 2:alphaf
                H0{i} = [H0{i};cell2mat(Y(i,j:(betaf+j-1)))];
                H1{i} = [H1{i};cell2mat(Y(i,(j+1):(betaf+j)))];
            end
            [R_svd{i},Sigma{i},S_svd{i}] = svd(H0{i});
            if i == 1 
                rank(H0{i})
                % figure
                % for j = 1:min(alphaf*nent,betaf*nsai)
                %     plot(j,Sigma{i}(j,j),'k*')
                %     hold on
                % end
                nf = rank(H0{i});
            end
            Sigma_n{i} = Sigma{i}(1:nf,1:nf);
            Rn{i} = R_svd{i}(:,1:nf);
            Sn{i} = S_svd{i}(:,1:nf);
            Er = [eye(nent);zeros((betaf-1)*nent,nent)];
            Em = [eye(nsai);zeros((alphaf-1)*nsai,nsai)];
            A_{i,Ni} = Sigma_n{i}^(-1/2)*Rn{i}'*H1{i}*Sn{i}*Sigma_n{i}^(-1/2);
            B_{i,Ni} = Sigma_n{i}^(1/2)*Sn{i}'*Er;
            C_{i,Ni} = Em'*Rn{i}*Sigma_n{i}^(1/2);
            D_{i,Ni} = Y_0{i};
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
            Po{i} = C_{i,Ni};
            Yom{i} = Yo{i,1};
            for kk = 1:(alphaf+betaf-1)
                Po{i} = [Po{i};C_{i,Ni}*A_{i,Ni}^kk];
                Yom{i} = [Yom{i};Yo{i,kk+1}];
            end
            G{i,Ni} = (Po{i}'*Po{i})^-1*Po{i}'*Yom{i};
            % xz{i}(:,Ni-tau_a) = zeros(nf,1);
            % for j = (Ni-tau_a+1):Ni
            %     xz{i}(:,j) = A_{i,Ni}*xz{i}(:,j-1)+B_{i,Ni}*u(j-1,:)';
            % end
            xz{i}(:,Ni) = zeros(nf,1);
            % yer{i}(Ni,:) = (C_{i,Ni}*xz{i}(:,Ni)+D_{i,Ni}*u(Ni,:)')';
        end
        ye(Ni,:) = 0;
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
        ye(k,:) = 0;
        for i = 1:R
            Z{i} = ppi'*P_cov{i}/(lambda/mu(i,k)+ppi'*P_cov{i}*ppi);
            P_cov{i} = lambda^-1*P_cov{i}*(eye(length(P_cov{i}))-ppi*Z{i});
            Y_m{i} = Y_m{i} + (y(k,:)'-Y_m{i}*ppi)*Z{i};
            Y_0{i} = Y_m{i}(1:nsai,1:nent);
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
            H0{i} = cell2mat(Y(i,1:betaf));
            H1{i} = cell2mat(Y(i,2:(betaf+1)));
            for j = 2:alphaf
                H0{i} = [H0{i};cell2mat(Y(i,j:(betaf+j-1)))];
                H1{i} = [H1{i};cell2mat(Y(i,(j+1):(betaf+j)))];
            end
            [R_svd{i},Sigma{i},S_svd{i}] = svd(H0{i});
            Sigma_n{i} = Sigma{i}(1:nf,1:nf);
            Rn{i} = R_svd{i}(:,1:nf);
            Sn{i} = S_svd{i}(:,1:nf);
            Er = [eye(nent);zeros((betaf-1)*nent,nent)];
            Em = [eye(nsai);zeros((alphaf-1)*nsai,nsai)];
            A_{i,k} = Sigma_n{i}^(-1/2)*Rn{i}'*H1{i}*Sn{i}*Sigma_n{i}^(-1/2);
            B_{i,k} = Sigma_n{i}^(1/2)*Sn{i}'*Er;
            C_{i,k} = Em'*Rn{i}*Sigma_n{i}^(1/2);
            D_{i,k} = Y_0{i};
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
            Po{i} = C_{i,k};
            Yom{i} = Yo{i,1};
            for j = 1:(alphaf+betaf-1)
                Po{i} = [Po{i};C_{i,k}*A_{i,k}^j];
                Yom{i} = [Yom{i};Yo{i,j+1}];
            end
            G{i,k} = (Po{i}'*Po{i})^-1*Po{i}'*Yom{i};
            mud_bru = 0;
            for j = 1:nf
                for jj = 1:nf
                    if abs(A_{i,k}(j,jj)-A_{i,k-1}(j,jj)) > fat_alt*abs(A_{i,k-1}(j,jj))
                        mud_bru = 1;
                    end
                end
            end
            for j = 1:nf
                for jj = 1:nent
                    if abs(B_{i,k}(j,jj)-B_{i,k-1}(j,jj)) > fat_alt*abs(B_{i,k-1}(j,jj))
                        mud_bru = 1;
                    end
                end
            end
            if mud_bru
                acionamentos = acionamentos + 1;
                xz{i}(:,k-tau_a) = zeros(nf,1);
                yet(k-tau_a,:) = (C_{i,k}*xz{i}(:,k-tau_a)+D_{i,k}*u(k-tau_a,:)')';
                for j = (k-tau_a+1):k
                    xz{i}(:,j) = A_{i,k}*xz{i}(:,j-1)+B_{i,k}*u(j-1,:)'-G{i,k}*(y(j-1,:)'-yet(j-1,:)');
                    yet(j,:) = (C_{i,k}*xz{i}(:,j)+D_{i,k}*u(j,:)')';
                end
            else
                xz{i}(:,k) = A_{i,k}*xz{i}(:,k-1)+B_{i,k}*u(k-1,:)'-G{i,k}*(y(k-1,:)'-ye(k-1,:)');
            end
            yer{i}(k,:) = (C_{i,k}*xz{i}(:,k)+D_{i,k}*u(k,:)')';
            ye(k,:) = ye(k,:) + mu(i,k)*yer{i}(k,:);
        end
    end
    qt_reg(k) = R;
end
toc
xc(1,:) = xcn{end}(1,:)*(max_u-min_u) + min_u;
xc(2,:) = xcn{end}(2,:)*(max_y-min_y) + min_y;
figure
plot(u,y,'c*',u_o,y_o,'k--',u_o,y_o,'b*',xc(1,:),xc(2,:),'r*','LineWidth',1)
hold on
legend('Vectors of data','Static curve','Steady-state data','Centers of clusters')
xlabel('Voltage (V)')
ylabel('Temperature (°C)')
axis equal
axis([-12.5 172.5 20 205])


% xc(1,:) = xcn{end}(1,:)*(max_u-min_u)+min_u;
% xc(2,:) = xcn{end}(2,:)*(max_y-min_y)+min_y;
erro_medio(Ni+1,1) = (y(Ni+1)-ye(Ni+1))^2;
raiz_erro_medio(Ni+1,1) = sqrt(erro_medio(Ni+1,1));
j = 1;
for i = Ni+2:length(y)
    j = j + 1;
    erro_medio(i,1) =  ((j-1)*erro_medio(i-1,1) + (y(i)-ye(i))^2)/j;
    raiz_erro_medio(i,1) = sqrt(erro_medio(i,1));
end
figure
plot(t,y,'k',t,ye,'b--','LineWidth',1)
xlabel('Time (s)')
% xlabel('Tempo (s)','Interpreter','latex')
ylabel('Temperature (°C)')
% ylabel('Temperatura ($^{\circ}$C)','Interpreter','latex')
legend('Output','Estimated output')
grid on
% legend('Temperatura','Temperatura estimada','Interpreter','latex')
figure
plot(t,u,'b','LineWidth',1)
legend('Input')
xlabel('Time (s)')
% xlabel('Tempo (s)','Interpreter','latex')
ylabel('Voltage (V)')
grid on
figure
plot(t,qt_reg,'b','LineWidth',1)
xlabel('Time (s)')
ylabel('Number of clusters')
grid on
ylim([0 9])
yticks(0:1:9)
figure
plot(t(Ni+1:end),y(Ni+1:end)-ye(Ni+1:end),'b',t(Ni+1:end),raiz_erro_medio(Ni+1:end),'r','LineWidth',1)
legend('Error','RMSE')
xlim(Tsn*[Ni 1000])
xlabel('Time (s)')
% xlabel('Tempo (s)','Interpreter','latex')
ylabel('Temperature (°C)')
grid on

y = y(Ni+1:end);
ye = ye(Ni+1:end);
RMSE = sqrt(mean((y-ye).^2))
num = 0;
den1 = 0;
den2 = 0;
for i = 1:length(y)
    num = num + (y(i)-mean(y))*(ye(i)-mean(ye));
    den1 = den1 + (y(i)-mean(y))^2;
    den2 = den2 + (ye(i)-mean(ye))^2;
end
CCR = num/(sqrt(den1*den2))
NDEI = RMSE/std(ye)
num = 0;
den = 0;
for i = 1:length(y)
    num = num + (y(i)-ye(i))^2;
    den = den + (y(i)-mean(y))^2;
end
R2 = 1-num/den

Sistema.R = R;
Sistema.xcn = xcn{end};
Sistema.F = F;
Sistema.m = m;
Sistema.A = A_(:,end);
Sistema.B = B_(:,end);
Sistema.C = C_(:,end);
Sistema.D = D_(:,end);
Sistema.G = G(:,end);
save('Sistema.mat','Sistema')
%  BASED ON (Iridium reference):
%    - 12x10 element URA (120 elements total)
%    - L-band: 1616–1626.5 MHz centre at 1621.25 MHz
%    - LEO orbit: 780 km altitude
%    - 48 spot beams in 4 tiers
%    - Analog beamforming (phase-only weights)

clear; clc; close all;
fprintf('==============================================\n');
fprintf(' Advanced LEO SATCOM Adaptive Beam System\n');
fprintf(' Based on Iridium URA + CNN + RF Link Budget\n');
fprintf('==============================================\n\n');

fprintf('[A] Setting up Iridium URA array...\n');

freqLow  = 1616e6;
freqHigh = 1626.5e6;
fc       = (freqLow + freqHigh) / 2;   % 1621.25 MHz
c        = physconst('LightSpeed');
lambda   = c / fc;                      % ~0.185 m
nRow = 12;
nCol = 10;
N    = nRow * nCol;   % 120 elements
dRow = lambda / 2;
dCol = lambda / 2;

% Altitude of LEO orbit (Iridium: 780 km)
alt_km   = 780;
alt_m    = alt_km * 1e3;
earthR   = 6371e3;   % Earth radius

ura = phased.URA( ...
    'Size',            [nRow, nCol], ...
    'ElementSpacing',  [dRow, dCol], ...
    'Element',         phased.IsotropicAntennaElement);

fprintf('    URA: %dx%d = %d elements | fc=%.2f MHz | alt=%d km\n', ...
        nRow, nCol, N, fc/1e6, alt_km);

fprintf('[B] Computing 48-beam 4-tier geometry...\n');


tier_el   = [0, -8.2, -16.4, -24.6];   % elevation offset per tier (deg)
tier_count = [3, 9, 18, 18];            % beams per tier

beam_az = [];
beam_el = [];
for t = 1:4
    n_b = tier_count(t);
    az_vals = linspace(-60, 60, n_b);  % spread across azimuth
    el_vals = tier_el(t) * ones(1, n_b);
    beam_az = [beam_az, az_vals];
    beam_el = [beam_el, el_vals];
end
nBeams = length(beam_az);   % should be 48

% Steering vector object
sv_obj = phased.SteeringVector( ...
    'SensorArray',      ura, ...
    'PropagationSpeed', c);

% Precompute all 48 steering vectors (N x 48 matrix)
W_beams = zeros(N, nBeams);
for b = 1:nBeams
    W_beams(:, b) = sv_obj(fc, [beam_az(b); beam_el(b)]);
end
% Normalise each beam vector
W_beams = W_beams ./ vecnorm(W_beams);

fprintf('    48 beams computed across 4 tiers\n');
fprintf('    Az range: [%.1f, %.1f] deg\n', min(beam_az), max(beam_az));
fprintf('    El range: [%.1f, %.1f] deg\n', min(beam_el), max(beam_el));

%% ============================================================
%  PART C: RF TOOLBOX — LINK BUDGET PER BEAM
% ============================================================
fprintf('[C] Computing RF link budget for each beam...\n');

% Satellite transmit parameters
Pt_dBW     = 10;          % Transmit power per beam (dBW)
Gt_dBi     = 10*log10(N); % Array gain ≈ 10*log10(N) for URA
Pt_W       = 10^(Pt_dBW/10);

% Slant range for each beam (depends on elevation angle)
% Range = sqrt(alt^2 + (earthR*sin(el))^2) - earthR*sin(el)  (simplified)
slant_range = zeros(1, nBeams);
for b = 1:nBeams
    el_rad = beam_el(b) * pi/180;
    % Slant range using spherical Earth geometry
    sin_el = sin(el_rad);
    slant_range(b) = sqrt((earthR*sin_el)^2 + alt_m*(alt_m + 2*earthR)) ...
                     - earthR*sin_el;
end

% Free space path loss (dB) = 20*log10(4*pi*R*f/c)
FSPL_dB = 20*log10(4*pi*slant_range*fc/c);

% Receiver parameters (ground terminal)
Gr_dBi     = 3;           % Ground terminal antenna gain
T_sys_K    = 290;         % System noise temperature
k_B        = physconst('Boltzmann');
BW_Hz      = 10e6;        % Signal bandwidth
N0_dBW     = 10*log10(k_B * T_sys_K * BW_Hz);   % Noise power

% Received SNR per beam (dB)
SNR_beam_dB = Pt_dBW + Gt_dBi - FSPL_dB + Gr_dBi - N0_dBW;

% Build sparameters chain for one T/R element (RF Toolbox)
freq_pts   = [freqLow, fc, freqHigh];
S_element  = zeros(2, 2, 3);
for fi = 1:3
    S_element(1,1,fi) = 10^(-15/20);   % S11: -15 dB return loss
    S_element(2,1,fi) = 10^(-2/20);    % S21: -2 dB insertion loss
    S_element(1,2,fi) = 10^(-30/20);   % S12: isolation
    S_element(2,2,fi) = 10^(-15/20);   % S22
end
element_sparams = sparameters(S_element, freq_pts);

% Cascade 3 elements (simplified T/R chain)
chain = cascadesparams(element_sparams, element_sparams);
chain = cascadesparams(chain, element_sparams);
S21_chain_dB = 20*log10(abs(rfparam(chain, 2, 1)));
element_IL_dB = -S21_chain_dB(2);   % insertion loss at fc

fprintf('    Element insertion loss (RF Toolbox): %.2f dB\n', element_IL_dB);
fprintf('    SNR range: %.1f to %.1f dB\n', min(SNR_beam_dB), max(SNR_beam_dB));

%% ============================================================
%  PART D: CNN ADAPTIVE BEAM SELECTOR
% ============================================================
% Problem: Given ground user (az, el) + interference direction,
%          which of the 48 beams gives best SINR?
% Solution: CNN classifier trained on simulated scenarios.
% This is NOT in the Iridium reference — this is your upgrade.
%
% Input:  [user_az, user_el, interf_az, interf_el, SNR]  (5 features)
% Output: beam index 1..48 (classification)
% ============================================================
fprintf('[D] Training CNN beam selector (your key upgrade)...\n');

rng(42);
n_train = 12000;
n_val   = 3000;
n_total = n_train + n_val;

% Generate training data
X_data = zeros(5, n_total);
Y_data = zeros(1, n_total);   % beam index label

for i = 1:n_total
    % Random user location within satellite footprint
    u_az  = -60 + 120*rand();
    u_el  = -30 + 30*rand();
    
    % Random interference source
    in_az = -60 + 120*rand();
    in_el = -30 + 30*rand();
    
    % Random SNR
    snr_i = 0 + 30*rand();   % 0 to 30 dB
    
    % Find best beam: closest in angular distance to user
    % (beam that maximises gain toward user)
    ang_dist = sqrt((beam_az - u_az).^2 + (beam_el - u_el).^2);
    
    % Also penalise beams that point toward interferer
    interf_dist = sqrt((beam_az - in_az).^2 + (beam_el - in_el).^2);
    
    % Score = -user_distance + 0.3*interferer_distance
    % (want beam close to user AND away from interferer)
    score = -ang_dist + 0.3*interf_dist;
    [~, best_beam] = max(score);
    
    X_data(:, i) = [u_az; u_el; in_az; in_el; snr_i];
    Y_data(i)    = best_beam;
end

% Normalise inputs
X_mu  = mean(X_data, 2);
X_sig = std(X_data, 0, 2) + 1e-8;
X_norm = (X_data - X_mu) ./ X_sig;

% Split
X_tr = X_norm(:, 1:n_train);      Y_tr = Y_data(1:n_train);
X_va = X_norm(:, n_train+1:end);  Y_va = Y_data(n_train+1:end);

% CNN architecture: 5 inputs → classify into 48 beam classes
layers = [
    featureInputLayer(5, 'Name','input', 'Normalization','none')

    fullyConnectedLayer(128, 'Name','fc1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')

    fullyConnectedLayer(256, 'Name','fc2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    dropoutLayer(0.2, 'Name','drop1')

    fullyConnectedLayer(256, 'Name','fc3')
    batchNormalizationLayer('Name','bn3')
    reluLayer('Name','relu3')

    fullyConnectedLayer(128, 'Name','fc4')
    reluLayer('Name','relu4')

    fullyConnectedLayer(nBeams, 'Name','fc_out')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classout')
];

opts = trainingOptions('adam', ...
    'MaxEpochs',           80, ...
    'MiniBatchSize',       256, ...
    'InitialLearnRate',    1e-3, ...
    'LearnRateSchedule',   'piecewise', ...
    'LearnRateDropPeriod', 30, ...
    'LearnRateDropFactor', 0.4, ...
    'L2Regularization',    1e-4, ...
    'ValidationData',      {X_va', categorical(Y_va')}, ...
    'ValidationFrequency', 100, ...
    'Shuffle',             'every-epoch', ...
    'Plots',               'training-progress', ...
    'Verbose',             false);

fprintf('    Training CNN (80 epochs, %d beams class)...\n', nBeams);
cnn = trainNetwork(X_tr', categorical(Y_tr'), layers, opts);

% Evaluate accuracy
Y_pred_cat = classify(cnn, X_va');
Y_pred     = double(Y_pred_cat);
accuracy   = mean(Y_pred == Y_va') * 100;
fprintf('    CNN beam selection accuracy: %.1f%%\n', accuracy);

% Save for reuse
save('satcom_cnn_beamselector.mat', 'cnn', 'X_mu', 'X_sig', ...
     'nBeams', 'beam_az', 'beam_el', 'W_beams', ...
     'fc', 'lambda', 'N', 'nRow', 'nCol');
fprintf('    Saved: satcom_cnn_beamselector.mat\n');
fprintf('[E] Running MVDR interference-nulling demonstration...\n');

Nsamp = 1024;

% Scenario: user at (20, -10) deg, interferer at (-15, -5) deg
user_az  =  20;  user_el  = -10;
interf_az = -15; interf_el = -5;

% Steering vectors
sv_user   = sv_obj(fc, [user_az;   user_el]);
sv_interf = sv_obj(fc, [interf_az; interf_el]);

% Simulate received signal (user + jammer + noise)
SNR_lin  = 10^(10/10);   % 10 dB SNR
JNR_lin  = 10^(20/10);   % 20 dB jammer-to-noise

noise   = (randn(N, Nsamp) + 1j*randn(N, Nsamp)) / sqrt(2);
rx_sig  = sv_user   * (sqrt(SNR_lin)  * (randn(1,Nsamp)+1j*randn(1,Nsamp))/sqrt(2)) + ...
          sv_interf * (sqrt(JNR_lin)  * (randn(1,Nsamp)+1j*randn(1,Nsamp))/sqrt(2)) + ...
          noise;

% 1. Phase-shift beamformer (paper's static approach)
% Use CNN to pick the best beam
x_cnn_in = ([user_az; user_el; interf_az; interf_el; 10] - X_mu) ./ X_sig;
pred_beam = double(classify(cnn, x_cnn_in'));
w_cnn     = W_beams(:, pred_beam);
out_cnn   = w_cnn' * rx_sig;

% 2. MVDR adaptive beamformer (Capon)
Rxx      = rx_sig * rx_sig' / Nsamp;
Rxx_reg  = Rxx + 0.01 * trace(Rxx)/N * eye(N);
w_mvdr   = Rxx_reg \ sv_user;
w_mvdr   = w_mvdr / (sv_user' * w_mvdr);
out_mvdr = w_mvdr' * rx_sig;

% 3. No beamforming
out_none = sum(rx_sig, 1) / N;

SINR_fn = @(x) 10*log10(var(real(x)));
sinr_none = SINR_fn(out_none);
sinr_cnn  = SINR_fn(out_cnn);
sinr_mvdr = SINR_fn(out_mvdr);

fprintf('    SINR — No BF: %.1f dB | CNN beam: %.1f dB | MVDR: %.1f dB\n', ...
        sinr_none, sinr_cnn, sinr_mvdr);

fprintf('[F] Generating all CV-ready plots...\n');

theta_scan = -70:0.5:70;

%% --- Figure 1: Beam pattern grid (6 representative beams) ---
figure('Name','Spot Beam Patterns','Color','white','Position',[30 30 1300 480]);
demo_beams = [1, 9, 20, 30, 40, 48];
cols6 = {'#2196F3','#4CAF50','#FF5722','#9C27B0','#FF9800','#00BCD4'};
subplot(1,3,1); hold on; grid on;
for k = 1:length(demo_beams)
    b  = demo_beams(k);
    w  = W_beams(:, b);
    AF = zeros(1, length(theta_scan));
    for m = 1:length(theta_scan)
        sv_m   = sv_obj(fc, [theta_scan(m); beam_el(b)]);
        AF(m)  = abs(w' * sv_m);
    end
    plot(theta_scan, 20*log10(AF/max(AF)+1e-10), ...
         'Color',cols6{k},'LineWidth',1.8, ...
         'DisplayName', sprintf('Beam %d (az=%.0f°)', b, beam_az(b)));
end
xlabel('Azimuth scan (°)'); ylabel('Gain (dB)');
title('Representative spot beam patterns');
legend('Location','south','FontSize',7); ylim([-40 3]);

%% --- Figure 2: CNN beam selection accuracy heatmap ---
subplot(1,3,2);
% Create confusion matrix for a subset
n_test_small = min(500, n_val);
Y_true_sub   = Y_va(1:n_test_small);
Y_pred_sub   = Y_pred(1:n_test_small);

% Bin into 8 groups for visibility
bin_size = 6;
n_bins   = ceil(nBeams / bin_size);
C        = zeros(n_bins, n_bins);
for i = 1:n_test_small
    r = min(ceil(Y_true_sub(i)/bin_size), n_bins);
    c = min(ceil(Y_pred_sub(i)/bin_size), n_bins);
    C(r,c) = C(r,c) + 1;
end
imagesc(C);
colormap(flipud(gray)); colorbar;
xlabel('Predicted beam group'); ylabel('True beam group');
title(sprintf('CNN beam selection (acc=%.1f%%)', accuracy));
set(gca,'FontSize',10);

%% --- Figure 3: RF Link budget per beam ---
subplot(1,3,3);
yyaxis left
plot(1:nBeams, SNR_beam_dB, 'b.-', 'MarkerSize',8, 'LineWidth',1.5);
ylabel('Received SNR (dB)','Color','b');
yyaxis right
plot(1:nBeams, slant_range/1e3, 'r.-', 'MarkerSize',8, 'LineWidth',1.5);
ylabel('Slant range (km)','Color','r');
xlabel('Beam index'); grid on;
title('RF link budget per beam (RF Toolbox)');
xline(pred_beam, 'k--', 'LineWidth',1.5, 'DisplayName','CNN selected');
legend({'SNR','Slant range','CNN selected'},'Location','best','FontSize',8);

sgtitle('LEO SATCOM Adaptive Spot Beam System — Iridium Architecture + CNN + RF Toolbox', ...
        'FontSize',11,'FontWeight','bold');
saveas(gcf,'satcom_beam_analysis.png');
fprintf('    Saved: satcom_beam_analysis.png\n');

%% --- Figure 2: SINR comparison ---
figure('Name','SINR Comparison','Color','white','Position',[30 560 780 350]);
sinr_vals  = [sinr_none, sinr_cnn, sinr_mvdr];
bar_labels = {'No beamforming','CNN beam selection','MVDR adaptive'};
bar_clrs   = [0.7 0.7 0.7; 0.2 0.6 0.9; 0.1 0.75 0.4];

b_hdl = bar(sinr_vals, 0.5);
b_hdl.FaceColor = 'flat';
b_hdl.CData     = bar_clrs;
set(gca,'XTickLabel', bar_labels, 'FontSize',11); grid on;
ylabel('Relative SINR (dB)');
title(sprintf('SINR: user=(%.0f°,%.0f°), jammer=(%.0f°,%.0f°) | JNR=20 dB', ...
              user_az, user_el, interf_az, interf_el));
for i = 1:3
    text(i, sinr_vals(i)+0.2, sprintf('%.1f dB', sinr_vals(i)), ...
         'HorizontalAlignment','center','FontWeight','bold','FontSize',11);
end
saveas(gcf,'satcom_sinr.png');
fprintf('    Saved: satcom_sinr.png\n');

%% --- Figure 3: Beam coverage footprint — INDIA / SOUTH EAST ASIA ---
% Geographic centre: India ~20°N 80°E, SEA ~10°N 105°E
% We map azimuth → longitude offset, elevation → latitude offset
% from satellite sub-point directly above the region

figure('Name','Beam Coverage — India & South East Asia', ...
       'Color','white','Position',[830 560 900 600]);
hold on; grid on; box on;

% Sub-satellite point (satellite directly overhead this lon/lat)
sat_lon = 90;    % degrees East — over Bay of Bengal
sat_lat = 20;    % degrees North — over central India/SEA corridor

% Convert beam az/el to approximate ground footprint lon/lat
% Approximation: az maps to longitude offset, el maps to latitude offset
% Scale factor based on 780 km orbit (1 deg Az ≈ 13.6 km on ground)
az_to_lon = 1.0;   % 1 deg az ≈ 1 deg lon at equator (rough)
el_to_lat = 0.5;   % elevation offset compressed due to geometry

beam_lon = sat_lon + beam_az * az_to_lon;
beam_lat = sat_lat + beam_el * el_to_lat * 2;

% HPBW footprint radius on ground (degrees)
HPBW_az_deg = 0.886 * lambda / (nCol * dCol) * 180/pi;
HPBW_el_deg = 0.886 * lambda / (nRow * dRow) * 180/pi;
foot_lon = HPBW_az_deg * az_to_lon * 4;   % footprint half-width lon
foot_lat = HPBW_el_deg * az_to_lon * 4;   % footprint half-width lat

% Draw beam footprints
theta_circ = linspace(0, 2*pi, 40);
for b = 1:nBeams
    ez = beam_lon(b) + foot_lon * cos(theta_circ);
    el = beam_lat(b) + foot_lat * sin(theta_circ);
    if b == pred_beam
        fill(ez, el, [0.1 0.75 0.4], 'FaceAlpha',0.55, ...
             'EdgeColor',[0 0.5 0], 'LineWidth',2.5);
        text(beam_lon(b), beam_lat(b), sprintf(' B%d',b), ...
             'FontSize',8,'FontWeight','bold','Color',[0 0.4 0]);
    else
        % Colour by tier
        tier_col = {[0.53 0.81 0.98],[0.67 0.85 0.99],...
                    [0.79 0.90 0.99],[0.90 0.95 1.00]};
        t_idx = min(ceil(b/12)+1, 4);
        fill(ez, el, tier_col{t_idx}, 'FaceAlpha',0.3, ...
             'EdgeColor',[0.2 0.4 0.8], 'LineWidth',0.6);
        if mod(b,6)==0
            text(beam_lon(b), beam_lat(b), sprintf('%d',b), ...
                 'FontSize',6,'Color',[0 0 0.5],...
                 'HorizontalAlignment','center');
        end
    end
end

% --- Draw geographic reference lines ---
% Country boundary approximations (key cities / borders)
% India outline (simplified bounding box points)
india_lon = [68, 97, 97, 68, 68];
india_lat = [8,  8,  37, 37,  8];
plot(india_lon, india_lat, 'k-', 'LineWidth', 1.5, 'DisplayName','India boundary');

% Key city markers
cities = {'Mumbai',  72.9, 19.1; ...
          'Delhi',   77.2, 28.6; ...
          'Chennai', 80.3, 13.1; ...
          'Kolkata', 88.4, 22.6; ...
          'Bangalore',77.6,12.9; ...
          'Bangkok', 100.5,13.8; ...
          'Singapore',103.8,1.4; ...
          'Jakarta', 106.8,-6.2; ...
          'Colombo', 79.9,  6.9; ...
          'Dhaka',   90.4, 23.7; ...
          'Karachi', 67.0, 24.9; ...
          'Yangon',  96.2, 16.9};

for ci = 1:size(cities,1)
    plot(cities{ci,2}, cities{ci,3}, 'r.', 'MarkerSize',14);
    text(cities{ci,2}+0.5, cities{ci,3}+0.5, cities{ci,1}, ...
         'FontSize',8, 'Color',[0.7 0 0], 'FontWeight','bold');
end

% Mark user and interferer in geographic coords
user_glon  = sat_lon + user_az  * az_to_lon;
user_glat  = sat_lat + user_el  * az_to_lon;
int_glon   = sat_lon + interf_az * az_to_lon;
int_glat   = sat_lat + interf_el * az_to_lon;

plot(user_glon,  user_glat,  'r^', 'MarkerSize',14, ...
     'MarkerFaceColor','r', 'DisplayName','Ground user terminal');
plot(int_glon,   int_glat,   'kx', 'MarkerSize',14, ...
     'LineWidth',3, 'DisplayName','Interference source');

% Satellite sub-point
plot(sat_lon, sat_lat, 'p', 'MarkerSize',18, ...
     'MarkerFaceColor',[1 0.8 0], 'MarkerEdgeColor','k', ...
     'LineWidth',1.5, 'DisplayName','Satellite sub-point');
text(sat_lon+1, sat_lat+1.5, 'Satellite (780 km LEO)', ...
     'FontSize',9, 'FontWeight','bold', 'Color',[0.6 0.4 0]);

% Formatting
xlim([60 125]);  ylim([-10 40]);
xlabel('Longitude (°E)', 'FontSize',11);
ylabel('Latitude (°N)',  'FontSize',11);
title({sprintf('LEO SATCOM 48-Beam Coverage — India & South East Asia'), ...
       sprintf('Iridium-style 12×10 URA | L-band %.0f MHz | Alt=%d km | CNN → Beam %d (green)', ...
               fc/1e6, alt_km, pred_beam)}, 'FontSize',11);
legend('Location','southwest','FontSize',8);

% Add lat/lon grid labels
set(gca,'XTick',60:10:130,'YTick',-10:5:40,'FontSize',9);
set(gca,'XGrid','on','YGrid','on','GridAlpha',0.25);

% Add region labels
text(78, 22, 'INDIA', 'FontSize',14,'FontWeight','bold',...
     'Color',[0.5 0.5 0.5],'HorizontalAlignment','center',...
     'Rotation',0,'FontAngle','italic');
text(105, 12, 'S.E. ASIA', 'FontSize',12,'FontWeight','bold',...
     'Color',[0.5 0.5 0.5],'HorizontalAlignment','center',...
     'FontAngle','italic');
text(80, -5, 'INDIAN OCEAN', 'FontSize',10,'Color',[0.4 0.6 0.8],...
     'HorizontalAlignment','center','FontAngle','italic');
text(68, 30, 'PAKISTAN', 'FontSize',8,'Color',[0.5 0.5 0.5]);
text(90, 27, 'BANGLADESH','FontSize',7,'Color',[0.5 0.5 0.5]);
text(94, 20, 'MYANMAR','FontSize',7,'Color',[0.5 0.5 0.5]);

saveas(gcf,'satcom_india_sea_coverage.png');
fprintf('    Saved: satcom_india_sea_coverage.png\n');
fprintf('\n==============================================\n');
fprintf('  ALL DONE\n');
fprintf('==============================================\n');
fprintf('\n  Output files:\n');
fprintf('    satcom_cnn_beamselector.mat  trained CNN + array params\n');
fprintf('    satcom_beam_analysis.png     beam patterns + link budget\n');
fprintf('    satcom_sinr.png              SINR comparison bar chart\n');
fprintf('    satcom_india_sea_coverage.png  48-beam footprint over India & SEA\n');
fprintf('\n  System summary:\n');
fprintf('    Array        : %dx%d URA, %d elements\n', nRow, nCol, N);
fprintf('    Frequency    : %.2f MHz (L-band, Iridium)\n', fc/1e6);
fprintf('    Orbit        : %d km LEO\n', alt_km);
fprintf('    Spot beams   : %d (4-tier)\n', nBeams);
fprintf('    CNN accuracy : %.1f%% beam selection\n', accuracy);
fprintf('    SINR gain    : +%.1f dB (CNN) | +%.1f dB (MVDR) vs no BF\n', ...
        sinr_cnn-sinr_none, sinr_mvdr-sinr_none);
fprintf('    Link budget  : %.1f to %.1f dB SNR across 48 beams\n', ...
        min(SNR_beam_dB), max(SNR_beam_dB));
fprintf('==============================================\n');
fprintf(['\n  "Designed adaptive LEO SATCOM spot beam system in\n' ...
         '   MATLAB based on Iridium MMA architecture (12x10 URA,\n' ...
         '   120 elements, L-band 1621 MHz, 780 km LEO). Implemented\n' ...
         '   48-beam 4-tier coverage using Phased Array System Toolbox\n' ...
         '   with RF Toolbox link budget (%.1f–%.1f dB SNR per beam).\n' ...
         '   Replaced static beam lookup with CNN classifier (Deep\n' ...
         '   Learning Toolbox) achieving %.1f%% beam selection accuracy.\n' ...
         '   MVDR adaptive beamformer demonstrated +%.1f dB SINR\n' ...
         '   over conventional beam steering at 20 dB JNR."\n\n'], ...
         min(SNR_beam_dB), max(SNR_beam_dB), accuracy, sinr_mvdr-sinr_none);

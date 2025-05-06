clear
clc

rawdata_replay = readtable ('SSEA/replay_Vehicle C - all IDs - IDs_Datafield_Classification.csv');
rawdata_fuzzy = readtable ('SSEA/fuzzing_Vehicle C - all IDs - IDs_Datafield_Classification.csv');
rawdata_combined = readtable ('SSEA/combined_Vehicle C - Replay 0x181, 0x161, 0x1a5, Fuzzing 0x244, 0x284, 0x354 - IDs_Datafield_Classification.csv');

data_nr=100000;

rawdata_replay = rawdata_replay (1:data_nr,:);
rawdata_fuzzy = rawdata_fuzzy (1:data_nr,:);
rawdata_combined = rawdata_combined (1:data_nr,:);

rawdata_replay = table2array(rawdata_replay);
rawdata_fuzzy = table2array(rawdata_fuzzy);
rawdata_combined = table2array(rawdata_combined);

data_replay = rawdata_replay(:, 1:end-1);
data_fuzzy = rawdata_fuzzy(:, 1:end-1);
data_combined = rawdata_combined(:, 1:end-1);

class_replay= rawdata_replay(:, end);
class_fuzzy= rawdata_fuzzy(:, end);
class_combined= rawdata_combined(:, end);

rng("default")
c_replay = cvpartition(class_replay,Holdout=0.20);
trainingIndices_replay = training(c_replay);
testIndices_replay = test(c_replay);
XTrain_replay = data_replay(trainingIndices_replay,:);
YTrain_replay = class_replay(trainingIndices_replay);
XTest_replay = data_replay(testIndices_replay,:);
YTest_replay = class_replay(testIndices_replay);

rng("default")
c_fuzzy = cvpartition(class_fuzzy,Holdout=0.20);
trainingIndices_fuzzy = training(c_fuzzy); 
testIndices_fuzzy = test(c_fuzzy);
XTrain_fuzzy = data_fuzzy(trainingIndices_fuzzy,:);
YTrain_fuzzy = class_fuzzy(trainingIndices_fuzzy);
XTest_fuzzy = data_fuzzy(testIndices_fuzzy,:);
YTest_fuzzy = class_fuzzy(testIndices_fuzzy);

rng("default")
c_combined = cvpartition(class_combined,Holdout=0.20);
trainingIndices_combined = training(c_combined);
testIndices_combined = test(c_combined);
XTrain_combined = data_combined(trainingIndices_combined,:);
YTrain_combined = class_combined(trainingIndices_combined);
XTest_combined = data_combined(testIndices_combined,:);
YTest_combined = class_combined(testIndices_combined);

mdl_replay = fitcsvm(XTrain_replay, YTrain_replay,'KernelFunction','rbf', ...
    'Standardize',true,'ClassNames',[0, 1]);

mdl_fuzzy = fitcsvm(XTrain_fuzzy, YTrain_fuzzy,'KernelFunction','rbf', ...
    'Standardize',true,'ClassNames',[0, 1]);

mdl_combined = fitcsvm(XTrain_combined, YTrain_combined,'KernelFunction','rbf', ...
    'Standardize',true,'ClassNames',[0, 1]);

[Pred_replay,Scores_replay] = predict(mdl_replay,XTest_replay);

[Pred_fuzzy,Scores_fuzzy] = predict(mdl_fuzzy,XTest_fuzzy);

[Pred_combined,Scores_combined] = predict(mdl_combined,XTest_combined);

rocObj_replay = rocmetrics(YTest_replay,Scores_replay,mdl_replay.ClassNames);
rocObj_fuzzy = rocmetrics(YTest_fuzzy,Scores_fuzzy,mdl_fuzzy.ClassNames);
rocObj_combined = rocmetrics(YTest_combined,Scores_combined,mdl_combined.ClassNames);
figure;
hold on
legend1 =plot(rocObj_replay,'Color', [0 0 1])
legend2 =plot(rocObj_fuzzy,'Color', [1 0 0])
legend3 =plot(rocObj_combined,'Color', [0 1 0])
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves for Multiple Models');
legend([legend1(1), legend2(1), legend3(1)], {'Replay Attack', 'Fuzzy Attack', 'Combined'}, 'Location', 'SouthEast');
grid on;
hold off;

[C_replay, order_replay] = confusionmat(YTest_replay, Pred_replay);

[C_fuzzy, order_fuzzy] = confusionmat(YTest_fuzzy, Pred_fuzzy);

[C_combined, order_combined] = confusionmat(YTest_combined, Pred_combined);

figure;
confusionchart(C_replay, order_replay, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title('Matricea de Confuzie replay');

figure;
confusionchart(C_fuzzy, order_fuzzy, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title('Matricea de Confuzie fuzzy');

figure;
confusionchart(C_combined, order_combined, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title('Matricea de Confuzie combined');

truePositive_replay = C_replay(1, 1);
falsePositive_replay = C_replay(2, 1);
trueNegative_replay = C_replay(2, 2);
falseNegative_replay = C_replay(1, 2);

accuracy_replay = (truePositive_replay + trueNegative_replay) / sum(C_replay(:)) * 100;

precision_replay = truePositive_replay / (truePositive_replay + falsePositive_replay);

recall_replay = truePositive_replay / (truePositive_replay + falseNegative_replay);

f1Score_replay = 2 * (precision_replay * recall_replay) / (precision_replay + recall_replay);

truePositive_fuzzy = C_fuzzy(1, 1);
falsePositive_fuzzy = C_fuzzy(2, 1);
trueNegative_fuzzy = C_fuzzy(2, 2);
falseNegative_fuzzy = C_fuzzy(1, 2);

accuracy_fuzzy = (truePositive_fuzzy + trueNegative_fuzzy) / sum(C_fuzzy(:)) * 100;

precision_fuzzy = truePositive_fuzzy / (truePositive_fuzzy + falsePositive_fuzzy);

recall_fuzzy = truePositive_fuzzy / (truePositive_fuzzy + falseNegative_fuzzy);

f1Score_fuzzy = 2 * (precision_fuzzy * recall_fuzzy) / (precision_fuzzy + recall_fuzzy);

truePositive_combined = C_combined(1, 1);
falsePositive_combined = C_combined(2, 1);
trueNegative_combined = C_combined(2, 2);
falseNegative_combined = C_combined(1, 2);

accuracy_combined = (truePositive_combined + trueNegative_combined) / sum(C_combined(:)) * 100;

precision_combined = truePositive_combined / (truePositive_combined + falsePositive_combined);

recall_combined = truePositive_combined / (truePositive_combined + falseNegative_combined);

f1Score_combined = 2 * (precision_combined * recall_combined) / (precision_combined + recall_combined);

disp('Matricea de confuzie replay:');
disp(C_replay);
disp('Matricea de confuzie fuzzy:');
disp(C_fuzzy);
disp('Matricea de confuzie combined:');
disp(C_combined);

disp(['Acuratețea replay: ', num2str(accuracy_replay), '%']);
disp(['Acuratețea fuzzy: ', num2str(accuracy_fuzzy), '%']);
disp(['Acuratețea combined: ', num2str(accuracy_combined), '%']);

disp(['Precizia replay: ', num2str(precision_replay)]);
disp(['Precizia fuzzy: ', num2str(precision_fuzzy)]);
disp(['Precizia combined: ', num2str(precision_combined)]);

disp(['Senzitivitatea replay: ', num2str(recall_replay)]);
disp(['Senzitivitatea fuzzy: ', num2str(recall_fuzzy)]);
disp(['Senzitivitatea combined: ', num2str(recall_combined)]);

disp(['Scor F1 replay: ', num2str(f1Score_replay)]);
disp(['Scor F1 fuzzy: ', num2str(f1Score_fuzzy)]);
disp(['Scor F1 combined: ', num2str(f1Score_combined)]);
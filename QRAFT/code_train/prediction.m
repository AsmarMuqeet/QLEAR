function pred = prediction(x)
    load('QRAFT.mat','mdT1');
    pred = predict(mdT1,x);
end
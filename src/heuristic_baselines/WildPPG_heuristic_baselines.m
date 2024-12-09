datadir = '../../data/';
participant_files = dir(fullfile(datadir, 'WildPPG_Part_*.mat'));
for fidx = 1:length(participant_files)
    p = load(strcat(datadir, "WildPPG_Part_l38.mat"));%participant_files(fidx).name));
    ppgdats = {};
    ecgdats = {};
    for loc = ["sternum", "head", "ankle", "wrist"]
        ecgdat.fs = double(p.sternum.ecg.fs);
        ppgdat.fs = double(p.(loc).ppg_g.fs);
        ecgdat.v = p.sternum.ecg.v.';
        ppgdat.v = p.(loc).ppg_g.v.';
        ppgdats{end+1} = ppgdat;
        ecgdats{end+1} = ecgdat;
    end
    data = struct("ppg", ppgdats, "ecg", ecgdats); % each location is treated like a seperate participant, ecg not used at this point
    proc_folder = 'processing_files';
    mkdir(proc_folder)
    fn = strcat(proc_folder, '/WildPPG_processing_Part_',p.id,'.mat');
    save(fn, "data")
    options.dataset_file = fn;
    assess_beat_detectors(fn, options)
    break
end


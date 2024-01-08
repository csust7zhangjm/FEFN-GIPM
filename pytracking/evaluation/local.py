from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = '/root/autodl-tmp/full_data/'
    settings.got_packed_results_path = '/root/code/TransT/pytracking/tracking_results/got_packed_results/'
    settings.got_reports_path = ''
    settings.lasot_path = '/root/autodl-tmp/LaSOT/LaSOT/LaSOTBenchmark'
    settings.network_path = '/root/autodl-tmp/hyf_save_eps/checkpoints/ltr/transt/transt/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = '/root/code/TransT/pytracking/result_plots/'
    settings.results_path = '/root/code/TransT/pytracking/tracking_results'    # Where to store tracking results
    settings.segmentation_path = '/root/code/TransT/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/root/autodl-tmp/TrackingNet'
    settings.uav_path = ''
    settings.vot_path = '/root/autodl-tmp/VOT2020/'
    settings.youtubevos_dir = ''

    return settings


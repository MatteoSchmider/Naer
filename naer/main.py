import naer

spike = naer.Spike(5, 10, 0.0, 0)
synapse = naer.Synapse(1.0)
my_lif_type = naer.Lif(v_reset=0.0, v_thresh=0.020, t_ref=2000, t_mem=20000)
my_lif_type.reflect()
my_lif_type.print_values()



import datetime
import math
import os
import gzip
import platform
import random
import re
import shutil
import subprocess
import time
import traceback
import sys


from config import NUMBER_CORES, RESULTS_FOLDER, SNIPER_CONFIG, SCRIPTS, ENABLE_HEARTBEATS
from resultlib.plot import create_plots

HERE = os.path.dirname(os.path.abspath(__file__))
SNIPER_BASE = os.path.dirname(HERE)
BENCHMARKS = os.path.join(SNIPER_BASE, 'benchmarks')
BATCH_START = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M')

# Define the default frequency values
DEFAULT_MIN_FREQ = 1.0  # GHz
DEFAULT_MAX_FREQ = 4.0  # GHz


def change_base_configuration(base_configuration):
    base_cfg = os.path.join(SNIPER_BASE, 'config/base.cfg')
    with open(base_cfg, 'r') as f:
        content = f.read()
    with open(base_cfg, 'w') as f:
        for line in content.splitlines():
            m = re.match('.*cfg:(!?)([a-zA-Z_\\.0-9]+)$', line)
            if m:
                inverted = m.group(1) == '!'
                include = inverted ^ (m.group(2) in base_configuration)
                included = line[0] != '#'
                if include and not included:
                    line = line[1:]
                elif not include and included:
                    line = '#' + line
            f.write(line)
            f.write('\n')


def prev_run_cleanup():
    '''Cleanup files potentially left over from aborted previous runs.'''

    pattern = r"^\d+\.hb.log$" # Heartbeat logs
    for f in os.listdir(BENCHMARKS):
        if not re.match(pattern, f):
            continue

        file_path = os.path.join(BENCHMARKS, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for f in os.listdir(BENCHMARKS):
        if ('output.' in f) or ('.264' in f) or ('poses.' in f) or ('app_mapping' in f) :
            os.remove(os.path.join(BENCHMARKS, f))
        

def save_output(base_configuration, benchmark, console_output, cpistack, started, ended):
    benchmark_text = benchmark
    if len(benchmark_text) > 100:
        benchmark_text = benchmark_text[:100] + '__etc'
    run = 'results_{}_{}_{}'.format(BATCH_START, '+'.join(base_configuration), benchmark_text)
    directory = os.path.join(RESULTS_FOLDER, run)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with gzip.open(os.path.join(directory, 'execution.log.gz'), 'w') as f:
        f.write(console_output.encode('utf-8'))
    with open(os.path.join(directory, 'executioninfo.txt'), 'w') as f:
        f.write('started:    {}\n'.format(started.strftime('%Y-%m-%d %H:%M:%S')))
        f.write('ended:      {}\n'.format(ended.strftime('%Y-%m-%d %H:%M:%S')))
        f.write('duration:   {}\n'.format(ended - started))
        f.write('host:       {}\n'.format(platform.node()))
        f.write('tasks:      {}\n'.format(benchmark))
    with open(os.path.join(directory, 'cpi-stack.txt'), 'wb') as f:
        f.write(cpistack)
    for f in ('sim.cfg',
              'sim.info',
              'sim.out',
              'cpi-stack.png',
              'sim.stats.sqlite3'):
        shutil.copy(os.path.join(BENCHMARKS, f), directory)
    for f in ('PeriodicPower.log',
              'PeriodicThermal.log',
              'PeriodicFrequency.log',
              'PeriodicVdd.log',
              'PeriodicCPIStack.log',
              'PeriodicRvalue.log'):
        with open(os.path.join(BENCHMARKS, f), 'rb') as f_in, gzip.open('{}.gz'.format(os.path.join(directory, f)), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    pattern = r"^\d+\.hb.log$" # Heartbeat logs
    for f in os.listdir(BENCHMARKS):
        if not re.match(pattern, f):
            continue
        shutil.copy(os.path.join(BENCHMARKS, f), directory)
    
    for f in os.listdir(BENCHMARKS):
        if 'output.' in f:
            shutil.copy(os.path.join(BENCHMARKS, f), directory)
        elif 'poses.' in f:
            shutil.copy(os.path.join(BENCHMARKS, f), directory)
        elif '.264' in f:
            shutil.copy(os.path.join(BENCHMARKS, f), directory)
        elif 'app_mapping.' in f:
            shutil.copy(os.path.join(BENCHMARKS, f), directory)

    create_plots(run)


def generate_core_config(base_configuration):
    """
    Generate per-core configuration strings based on the base_configuration entries.
    This supports both symmetric and asymmetric DVFS configurations.
    
    Returns:
        A tuple of (dvfs_config, core_freq_config) where:
        - dvfs_config: Command-line parameters for the DVFS settings
        - core_freq_config: Dictionary mapping core numbers to frequencies for per-core overrides
    """
    # Default to empty configurations
    dvfs_config = ""
    core_freq_config = {}
    
    # Check if we have per-core frequency settings (asymmetric DVFS)
    core_freqs = []
    for cfg in base_configuration:
        # Match patterns like '4.0GHz', '1.0GHz', etc.
        if cfg.endswith('GHz') and ':' not in cfg:
            try:
                freq = float(cfg.replace('GHz', ''))
                # This is a global frequency setting
                dvfs_config += " -g perf_model/core/frequency=%s" % freq
            except ValueError:
                pass
        
        # Match patterns like 'core0:4.0GHz', 'core1:1.0GHz', etc.
        elif 'core' in cfg and ':' in cfg and cfg.endswith('GHz'):
            try:
                core_part, freq_part = cfg.split(':')
                core_num = int(core_part.replace('core', ''))
                freq = float(freq_part.replace('GHz', ''))
                
                # Add to our per-core frequency mapping
                core_freq_config[core_num] = freq
                core_freqs.append(freq)
            except (ValueError, IndexError):
                pass
    
    # If we have per-core frequencies, add them to the configuration
    for core_num, freq in core_freq_config.items():
        dvfs_config += " -g perf_model/core/%d/frequency=%s" % (core_num, freq)
    
    # Enable migration configuration if needed
    migration_enabled = False
    migration_epoch = "1000000"  # Default migration epoch (1ms)
    
    if "migrate10ms" in base_configuration:
        migration_epoch = "10000000"  # 10ms
        migration_enabled = True
    elif "migrate100us" in base_configuration:
        migration_epoch = "100000"  # 100Î¼s
        migration_enabled = True
    elif "coldestCore" in base_configuration:
        migration_enabled = True
    
    if migration_enabled:
        dvfs_config += " -g scheduler/open/migration/logic=coldestCore"
        dvfs_config += " -g scheduler/open/migration/epoch=%s" % migration_epoch
    else:
        dvfs_config += " -g scheduler/open/migration/logic=off"
    
    # Configure DVFS logic based on configuration
    dvfs_logic = "off"
    if "maxFreq" in base_configuration:
        dvfs_logic = "maxFreq"
    elif "ondemand" in base_configuration:
        dvfs_logic = "ondemand"
    elif "testStaticPower" in base_configuration:
        dvfs_logic = "testStaticPower"
    
    # Add DVFS logic configuration
    dvfs_config += " -g scheduler/open/dvfs/logic=%s" % dvfs_logic
    
    # Configure DVFS epoch (how often to adjust frequency)
    dvfs_epoch = "1000000"  # Default is 1ms
    if "fastDVFS" in base_configuration:
        dvfs_epoch = "100000"  # 100Î¼s
    elif "mediumDVFS" in base_configuration:
        dvfs_epoch = "250000"  # 250Î¼s
    elif "slowDVFS" in base_configuration:
        dvfs_epoch = "1000000"  # 1ms
    
    dvfs_config += " -g scheduler/open/dvfs/dvfs_epoch=%s" % dvfs_epoch
    dvfs_config += " -g dvfs/transition_latency=2000"  # Add realistic transition latency
    
    # Add DVFS min/max frequency settings
    min_freq = DEFAULT_MIN_FREQ
    max_freq = DEFAULT_MAX_FREQ
    
    # Override with any explicit min/max freq in base_configuration
    for cfg in base_configuration:
        if cfg.startswith("min_freq="):
            try:
                min_freq = float(cfg.split("=")[1])
            except (ValueError, IndexError):
                pass
        elif cfg.startswith("max_freq="):
            try:
                max_freq = float(cfg.split("=")[1])
            except (ValueError, IndexError):
                pass
    
    dvfs_config += " -g scheduler/open/dvfs/min_frequency=%s" % min_freq
    dvfs_config += " -g scheduler/open/dvfs/max_frequency=%s" % max_freq
    
    return dvfs_config, core_freq_config

def run(base_configuration, benchmark, ignore_error=False, perforation_script: str = None):
    """
    Run a simulation with the given base configuration and benchmark.
    
    Args:
        base_configuration: List of configuration options
        benchmark: The benchmark to run
        ignore_error: Whether to ignore errors during execution
        perforation_script: Optional perforation script
    """
    print('running {} with configuration {}'.format(benchmark, '+'.join(base_configuration)))
    started = datetime.datetime.now()
    
    # Apply configuration changes to base.cfg
    change_base_configuration(base_configuration)
    
    # Clean up from previous runs
    prev_run_cleanup()
    
    # Generate DVFS and per-core frequency configurations
    dvfs_config, core_freq_config = generate_core_config(base_configuration)
    
    # Prepare benchmark options
    benchmark_options = []
    if ENABLE_HEARTBEATS == True:
        benchmark_options.append('enable_heartbeats')
        benchmark_options.append('hb_results_dir=%s' % BENCHMARKS)

    # Configure logging interval
    periodicPower = 1000000  # Default is 1ms 
    if 'mediumDVFS' in base_configuration:
        periodicPower = 250000  # 250Î¼s
    if 'fastDVFS' in base_configuration:
        periodicPower = 100000  # 100Î¼s

    # Configure perforation script if needed
    if not perforation_script:
        perforation_script = 'magic_perforation_rate:' 
   
    # Build Sniper command arguments
    args = '-n {number_cores} -c {config} --benchmarks={benchmark} --no-roi --sim-end=last -senergystats:{periodic} -speriodic-power:{periodic}{script}{perforation}{benchmark_options}{dvfs_config}' \
        .format(number_cores=NUMBER_CORES,
                config=SNIPER_CONFIG,
                benchmark=benchmark,
                periodic=periodicPower,
                script= ''.join([' -s' + s for s in SCRIPTS]),
                perforation=' -s'+perforation_script,
                benchmark_options=''.join([' -B ' + opt for opt in benchmark_options]),
                dvfs_config=dvfs_config)
    
    console_output = ''

    # Print configuration details for debugging
    print("Command arguments:", args)
    if core_freq_config:
        print("Per-core frequency configuration:", core_freq_config)
    
    # Run the simulator
    run_sniper = os.path.join(BENCHMARKS, 'run-sniper')
    p = subprocess.Popen([run_sniper] + args.split(' '), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, cwd=BENCHMARKS)
    with p.stdout:
        for line in iter(p.stdout.readline, b''):
            linestr = line.decode('utf-8')
            console_output += linestr
            print(linestr, end='')

    p.wait()

    # Generate CPI stack visualization
    try:
        cpistack = subprocess.check_output(['python', os.path.join(SNIPER_BASE, 'tools/cpistack.py')], cwd=BENCHMARKS)
    except:
        if ignore_error:
            cpistack = b''
        else:
            raise

    ended = datetime.datetime.now()

    # Save results
    save_output(base_configuration, benchmark, console_output, cpistack, started, ended)

    if p.returncode != 0:
        raise Exception('return code != 0')


def try_run(base_configuration, benchmark, ignore_error=False):
    try:
        run(base_configuration, benchmark, ignore_error=ignore_error)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        for i in range(4):
            print('#' * 80)
        #print(e)
        print(traceback.format_exc())
        for i in range(4):
            print('#' * 80)
        input('Please press enter...')


class Infeasible(Exception):
    pass


def get_instance(benchmark, parallelism, input_set='small'):
    threads = {
        'parsec-blackscholes': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'parsec-bodytrack': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'parsec-canneal': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'parsec-dedup': [4, 7, 10, 13, 16],
        'parsec-fluidanimate': [2, 3, 0, 5, 0, 0, 0, 9],
        'parsec-streamcluster': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'parsec-swaptions': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'parsec-x264': [1, 3, 4, 5, 6, 7, 8, 9],
        'splash2-barnes': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'splash2-cholesky': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'splash2-fft': [1, 2, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16],
        'splash2-fmm': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'splash2-lu.cont': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'splash2-lu.ncont': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'splash2-ocean.cont': [1, 2, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16],
        'splash2-ocean.ncont': [1, 2, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16],
        'splash2-radiosity': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'splash2-radix': [1, 2, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16],
        'splash2-raytrace': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'splash2-water.nsq': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        'splash2-water.sp': [1, 2, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 16],  # other parallelism values run but are suboptimal -> don't allow in the first place
    }
    
    ps = threads[benchmark]
    if parallelism <= 0 or parallelism not in ps:
        raise Infeasible()
    p = ps.index(parallelism) + 1

    if benchmark.startswith('parsec') and not input_set.startswith('sim'):
        input_set = 'sim' + input_set

    return '{}-{}-{}'.format(benchmark, input_set, p)


def get_feasible_parallelisms(benchmark):
    feasible = []
    for p in range(1, 16+1):
        try:
            get_instance(benchmark, p)
            feasible.append(p)
        except Infeasible:
            pass
    return feasible


def get_workload(benchmark, cores, parallelism=None, number_tasks=None, input_set='small'):
    if parallelism is not None:
        number_tasks = math.floor(cores / parallelism)
        return get_workload(benchmark, cores, number_tasks=number_tasks, input_set=input_set)
    elif number_tasks is not None:
        if number_tasks == 0:
            if cores == 0:
                return []
            else:
                raise Infeasible()
        else:
            parallelism = math.ceil(cores / number_tasks)
            for p in reversed(range(1, min(cores, parallelism) + 1)):
                try:
                    b = get_instance(benchmark, p, input_set=input_set)
                    return [b] + get_workload(benchmark, cores - p, number_tasks=number_tasks-1, input_set=input_set)
                except Infeasible:
                    pass
            raise Infeasible()
    else:
        raise Exception('either parallelism or number_tasks needs to be set')


def example():
    for benchmark in (
                      'parsec-blackscholes',
                      #'parsec-bodytrack',
                      #'parsec-canneal',
                      #'parsec-dedup',
                      #'parsec-ferret'
                      #'parsec-fluidanimate',
                      #'parsec-streamcluster',
                      #'parsec-swaptions',
                      #'parsec-x264',
                      #'splash2-barnes',
                      #'splash2-fmm',
                      #'splash2-ocean.cont',
                      #'splash2-ocean.ncont',
                      #'splash2-radiosity',
                      #'splash2-raytrace',
                      #'splash2-water.nsq',
                      #'splash2-water.sp',
                      #'splash2-cholesky',
                      #'splash2-fft',
                      #'splash2-lu.cont',
                      #'splash2-lu.ncont',
                      #'splash2-radix',
                      ):

        min_parallelism = get_feasible_parallelisms(benchmark)[0]
        max_parallelism = get_feasible_parallelisms(benchmark)[-1]
        for freq in (1, 2):
            #for parallelism in (max_parallelism,):
            for parallelism in (3, ):
                # you can also use try_run instead
                run(['{:.1f}GHz'.format(freq), 'maxFreq', 'slowDVFS'], get_instance(benchmark, parallelism, input_set='simsmall'))

def example_symmetric_perforation():
    for benchmark in (
                      'parsec-blackscholes',
                      #'parsec-bodytrack',
                      #'parsec-streamcluster',
                      #'parsec-swaptions',
                      #'parsec-x264',
                      #'parsec-canneal',
                    ):

        min_parallelism = get_feasible_parallelisms(benchmark)[0]
        max_parallelism = get_feasible_parallelisms(benchmark)[-1]

        perforation_rate = str(50)
        for freq in (4, ):
            for parallelism in (4,):
                run(['{:.1f}GHz'.format(freq), 'maxFreq', 'slowDVFS'], get_instance(benchmark, parallelism, input_set='simsmall'), 
                    perforation_script="magic_perforation_rate:%s" % perforation_rate )

def example_asymmetric_perforation():
    for benchmark in (
                        ("parsec-blackscholes", 1),
                        #("parsec-bodytrack", 6),
                        #("parsec-streamcluster", 2),
                        #("parsec-swaptions", 2),
                        #("parsec-x264", 6),
                        #("parsec-canneal", 3),
                    ):
    
        loop_rates = [  str(i*10) for i in range(benchmark[1]) ]

        min_parallelism = get_feasible_parallelisms(benchmark[0])[0]
        max_parallelism = get_feasible_parallelisms(benchmark[0])[-1]
        for freq in (4, ):
            for parallelism in (4,):
                run(['{:.1f}GHz'.format(freq), 'maxFreq', 'slowDVFS'], get_instance(benchmark[0], parallelism, input_set='simsmall'), 
                    perforation_script='magic_perforation_rate:%s' % ','.join(loop_rates))


def multi_program():
    # In this example, two instances of blackscholes will be scheduled.
    # By setting the scheduler/open/arrivalRate base.cfg parameter to 2, the
    # tasks can be set to arrive at the same time.

    input_set = 'simsmall'
    base_configuration = ['4.0GHz', "maxFreq"]
    benchmark_set = (
        'parsec-blackscholes',
        'parsec-x264',
    )

    if ENABLE_HEARTBEATS == True:
        base_configuration.append('hb_enabled')

    benchmarks = ''
    for i, benchmark in enumerate(benchmark_set):
        min_parallelism = get_feasible_parallelisms(benchmark)[0]
        if i != 0:
            benchmarks = benchmarks + ',' + get_instance(benchmark, min_parallelism, input_set)
        else:
            benchmarks = benchmarks + get_instance(benchmark, min_parallelism, input_set)

    run(base_configuration, benchmarks)

    
def test_static_power():
    run(['4.0GHz', 'testStaticPower', 'slowDVFS'], get_instance('parsec-blackscholes', 3, input_set='simsmall'))


def ondemand_demo():
    # run([â€™{:.1f}GHzâ€™.format(4), â€™ondemandâ€™, â€™fastDVFSâ€™], get_instance(â€™parsecblackscholesâ€™, 3, input_set=â€™simsmallâ€™))
    run(['{:.1f}GHz'.format(4), 'ondemand', 'fastDVFS'], get_instance('parsec-blackscholes', 3, input_set='simsmall'))


def coldestcore_demo():
    run(['{:.1f}GHz'.format(2.4), 'maxFreq', 'slowDVFS', 'coldestCore'], get_instance('parsec-blackscholes', 3, input_set='simsmall'))


def asg2_multi_threading_experiments():
    """
    Run multi-threading experiments with 1-4 threads for each benchmark.
    This tests how performance, power, and temperature scale with thread count.
    """
    # Base configuration with DVFS settings
    base_cfg = ['4.0GHz', 'maxFreq', 'slowDVFS', 'testStaticPower', 'dvfs_enabled']
    
    # Benchmarks to test
    benchmarks = ('parsec-blackscholes', 'parsec-streamcluster')
    
    # Thread counts to test (1-4 threads)
    thread_counts = range(1, 5)
    
    for benchmark in benchmarks:
        # Get feasible thread counts for this benchmark
        feasible_threads = get_feasible_parallelisms(benchmark)
        
        # Test each thread count
        for threads in thread_counts:
            # Skip if this thread count isn't feasible for this benchmark
            if threads not in feasible_threads:
                print("Skipping {} with {} threads (not feasible)".format(benchmark, threads))
                continue
                
            # Create benchmark instance with specified thread count
            inst = get_instance(benchmark, threads, input_set='simsmall')
            
            print("[MULT-THR] {benchmark} @ {threads} threads â†’ {inst}".format(
                benchmark=benchmark, threads=threads, inst=inst))
            
            # Run the experiment
            run(base_cfg, inst)


def asg2_symmetric_dvfs_experiments():
    """
    Run symmetric DVFS experiments with different frequencies.
    All cores run at the same frequency for each experiment.
    """
    # Benchmarks to test
    benches = ('parsec-blackscholes', 'parsec-streamcluster')
    
    # Frequencies to test (in GHz)
    # Full range: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    freqs = [1.0, 2.0, 3.0, 4.0]  # Using 1.0, 2.0, 3.0, 4.0 GHz for clear stepping
    
    for benchmark in benches:
        # Use 3 threads on 4 cores to allow for some thermal headroom
        inst = get_instance(benchmark, 3, input_set='simsmall')
        
        for f in freqs:
            # Configure with frequency and DVFS settings
            base_cfg = ["{0}GHz".format(f), 'maxFreq', 'testStaticPower', 'slowDVFS', 'dvfs_enabled']
            
            print("[SYM-DVFS] {benchmark} @ {f}GHz â†’ {inst}".format(
                benchmark=benchmark, f=f, inst=inst))
            
            # Run the experiment
            run(base_cfg, inst)


def asg2_asymmetric_dvfs_experiments():
    """
    Run experiments with asymmetric DVFS settings (different frequencies per core).
    
    This function creates multiple asymmetric frequency patterns:
    - 1H-3L: One core at high frequency (4.0GHz), three cores at low frequency (1.0GHz)
    - 2H-2L: Two cores at high frequency, two cores at low frequency
    - 3H-1L: Three cores at high frequency, one core at low frequency
    """
    benches = ('parsec-blackscholes', 'parsec-streamcluster')
    
    # Define asymmetric DVFS patterns
    patterns = {
        '1H-3L': ['core0:4.0GHz', 'core1:1.0GHz', 'core2:1.0GHz', 'core3:1.0GHz'],
        '2H-2L': ['core0:4.0GHz', 'core1:4.0GHz', 'core2:1.0GHz', 'core3:1.0GHz'],
        '3H-1L': ['core0:4.0GHz', 'core1:4.0GHz', 'core2:4.0GHz', 'core3:1.0GHz'],
    }
    
    for benchmark in benches:
        inst = get_instance(benchmark, 4, input_set='simsmall')
        for name, freq_configs in patterns.items():
            # Add DVFS and other configuration options
            base_cfg = freq_configs + ['maxFreq', 'slowDVFS', 'testStaticPower']
            
            print("[AsymDVFS:{name}] {benchmark} â†’ {inst}".format(
                name=name, benchmark=benchmark, inst=inst))
            
            run(base_cfg, inst)


def asg2_thread_migration_experiments():
    """
    Run experiments with different thread migration strategies:
    - static: No migration (baseline)
    - coldestCore_1ms: Migrate to coldest core with 1ms epoch
    - coldestCore_10ms: Migrate to coldest core with 10ms epoch
    - coldestCore_100us: Migrate to coldest core with 100Î¼s epoch
    """
    benches = ('parsec-blackscholes', 'parsec-streamcluster')
    
    # Define migration policies with their config flags
    policies = {
        'static': [],  # No migration (baseline)
        'coldestCore_1ms': ['coldestCore'],  # Default 1ms epoch
        'coldestCore_10ms': ['coldestCore', 'migrate10ms'],  # 10ms epoch
        'coldestCore_100us': ['coldestCore', 'migrate100us']  # 100Î¼s epoch
    }
    
    for benchmark in benches:
        inst = get_instance(benchmark, 3, input_set='simsmall')  # Using 3 threads on 4 cores
        for name, extra in policies.items():
            # Add frequency and DVFS settings
            base_cfg = ['4.0GHz', 'maxFreq', 'slowDVFS', 'testStaticPower', 'dvfs_enabled'] + extra
            
            print("[Migration:{name}] {benchmark} â†’ {inst}".format(
                name=name, benchmark=benchmark, inst=inst))
            
            run(base_cfg, inst)


def asg2_multiprogramming_experiments():
    """
    Run multi-programming experiments with two benchmarks running concurrently.
    This tests how the system handles running different benchmarks simultaneously.
    
    Various thread splits are tested:
    - 1+3: blackscholes (1 thread) + streamcluster (3 threads)
    - 2+2: blackscholes (2 threads) + streamcluster (2 threads)
    - 3+1: blackscholes (3 threads) + streamcluster (1 thread)
    """
    # Thread distribution splits to test
    splits = [
        (1, 3),  # blackscholes: 1 thread, streamcluster: 3 threads
        (2, 2),  # blackscholes: 2 threads, streamcluster: 2 threads
        (3, 1),  # blackscholes: 3 threads, streamcluster: 1 thread
    ]
    
    # Test each thread distribution
    for a_threads, b_threads in splits:
        # Check if this thread count is feasible for both benchmarks
        try:
            inst_a = get_instance('parsec-blackscholes', a_threads, input_set='simsmall')
            inst_b = get_instance('parsec-streamcluster', b_threads, input_set='simsmall')
            
            # Combine benchmarks with comma separator for multi-programming
            bench_str = inst_a + ',' + inst_b
            
            # Configure with frequency and DVFS settings
            base_cfg = ['4.0GHz', 'maxFreq', 'slowDVFS', 'testStaticPower', 'dvfs_enabled']
            
            print("[MP:{a_threads}+{b_threads}] BSâ†’{inst_a}  SCâ†’{inst_b}".format(
                a_threads=a_threads, b_threads=b_threads, inst_a=inst_a, inst_b=inst_b))
            
            # For multi-programming we need to set the arrival rate
            # Set arrivalRate=2 to launch both applications at the same time
            base_cfg.append('arrivalRate=2')
            
            # Run the experiment
            run(base_cfg, bench_str)
        except Infeasible:
            print("Skipping multiprogramming with threads {}+{} (not feasible)".format(a_threads, b_threads))
            continue


def main():
    """
    Main function to run experiments.
    Uncomment the experiment type you want to run.
    """
    # Multi-threading experiments (1-4 threads)
    # asg2_multi_threading_experiments()
    
    # Symmetric DVFS experiments (all cores at the same frequency)
    # asg2_symmetric_dvfs_experiments()
    
    # Asymmetric DVFS experiments (different frequencies per core)
    # asg2_asymmetric_dvfs_experiments()
    
    # Thread migration experiments (different migration strategies)
    # asg2_thread_migration_experiments()
    
    # Multi-programming experiments (multiple benchmarks concurrently)
    # asg2_multiprogramming_experiments()
    
    # Run all experiment types in sequence
    print("Running all experiment types...")
    asg2_multi_threading_experiments()
    asg2_symmetric_dvfs_experiments()
    asg2_asymmetric_dvfs_experiments()
    asg2_thread_migration_experiments()
    asg2_multiprogramming_experiments()
    print("All experiments completed.")


if __name__ == '__main__':
    main()

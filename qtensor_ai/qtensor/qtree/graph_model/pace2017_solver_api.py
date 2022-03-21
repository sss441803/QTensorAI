"""
This module implements API to the PACE 2017
treewidth solvers.
Node numbering starts from 1 and nodes are *consequtive*
integers!
"""
import re
import time
import subprocess
import threading
import sys

ENCODING = sys.getdefaultencoding()


def run_exact_solver(data, command="tw-exact", cwd=None,
                     extra_args=None):
    """
    Runs the exact solver and collects its output

    Parameters
    ----------
    data : str
         Path to the input file
    command : str, optional
         Deafults to "tamaki_tw-exact"
    extra_args : str, optional
         Optional commands to the solver

    Returns
    -------
    output : str
         Process output
    """
    sh = command + " "
    if extra_args is not None:
        sh += extra_args

    process = subprocess.Popen(
        sh.split(), cwd=cwd, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE)
    process.stdin.write(data.encode(ENCODING))
    output, error = process.communicate()
    if error:
        raise ValueError(error)

    return output.decode(ENCODING)

def run_heuristic_solver_interactive(data, callback,
                                     callback_delay=0.5,
                                     command='./tw-heuristic',
                                     cwd=None,
                                     extra_args=None):
    """
    Runs the heuristic tamaki solver and allows to interactively process output

    Requires sarge to be installed

    Tamaki                          Python
    |                               |
    | <-- start---------------------|
    |                               |
    |-> if exist new line data -->---callback(info), wait(callback_delay)
    |                               |
    | <-- send interrupt <------- if callback raised
    |                               |
    |-> output solution *format2 -->| process solution
    |
    |terminate


    Parameters
    ----------
    data : str
        String with gr representation of a graph
    callback: callable
        Function to handle new info data. Info format: (time, tw).
        Should handle [None, None] properly. Raise to stop iteration
    callback_delay : float
        Waiting time between checking for output
    command : str, optional
        Deafults to "./tw-heuristic"
    extra_args : str, optional
        Optional commands to the solver



    Returns
    -------
    output : str
         Process output
    """
    sh = command + " "
    if extra_args is not None:
        sh += extra_args

    width_pattern = re.compile('^c width =(?P<width>.+)')
    time_pattern = re.compile('^c time =(?P<time>.+) ms')
    import sarge
    p = sarge.Command(sh.split(), cwd=cwd,
                      stdout=sarge.Capture(),
                      stderr=sarge.Capture()
                     )
    update_info = [None, None]
    p.run(input=data, async_=True)
    try:
        while True:

            """ This wierd order is to skip sleep in case of update """
            try:
                callback(update_info)
            except Exception as e:
                print(f'Exception: {e}. Stoppnig tamaki', file=sys.stderr)
                break

            err = p.stderr.read().decode()
            if err:
                raise Exception("Java Error:\n"+err)
            line = p.stdout.readline().decode()
            maybe_width = width_pattern.search(line)
            if maybe_width:
                width = int(maybe_width.group('width'))
                update_info[1] = width
            line = p.stdout.readline().decode()

            maybe_time = time_pattern.search(line)
            if maybe_time:
                time_ = int(maybe_time.group('time'))
                update_info[0] = time_
                continue

            time.sleep(callback_delay)
    except BaseException as e:
        print('Stopping tamaki', file=sys.stderr)
        raise
    finally:
        p.terminate()
    p.wait()
    data = p.stdout.read().decode()
    return data



def run_heuristic_solver(data, wait_time=1,
                         command="./tw-heuristic", cwd=None,
                         extra_args=None):
    """
    Runs the heuristic tamakisolver and collects its output

    Parameters
    ----------
    data : str
        String with gr representation of a graph
    wait_time : float
         Waiting time in seconds
    command : str, optional
         Deafults to "./tw-heuristic"
    extra_args : str, optional
         Optional commands to the solver

    Returns
    -------
    output : str
         Process output
    """
    sh = command + " "
    if extra_args is not None:
        sh += extra_args

    process = subprocess.Popen(
        sh.split(), cwd=cwd, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE)

    def terminate_process():
        process.send_signal(subprocess.signal.SIGTERM)

    timer = threading.Timer(
        wait_time, terminate_process)
    try:
        timer.start()
        process.stdin.write(data.encode(ENCODING))
        output, error = process.communicate()
    finally:
        timer.cancel()

    if error:
        raise ValueError(error)

    return output.decode(ENCODING)


def test_run_heuristic():
    d = run_heuristic_solver('', wait_time=1,
                             command="echo 42")
    assert(int(d) == 42)

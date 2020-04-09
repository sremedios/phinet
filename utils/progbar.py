import sys
import time

def show_progbar(
        cur_epoch, 
        total_epochs, 
        batch_size,
        cur_step, 
        num_instances, 
        loss, 
        acc,
        color_code,
        elapsed_time,
        progbar_length=10,
        ):

    # first clear last line
    ERASE_LINE = '\x1b[2K'
    sys.stdout.write(ERASE_LINE)

    # write template
    TEMPLATE = ("\r{}Epoch {}/{} [{:{}<{}}] "
               "Loss: {:.4f} "
               "Acc: {:.2%} "
               " |  "
               "{:.2f}s / it | "
               "ETA: {:>3.1f}s | "
               "Total: {:>3.1f}s"
               "\033[0;0m")

    # calculate eta
    num_seen = num_instances - (cur_step * batch_size) 
    eta = num_seen / batch_size * elapsed_time
    if eta < 0:
        eta = 0

    # calculate total
    total = min(num_instances, cur_step * batch_size) / batch_size * elapsed_time

    current_progress = "=" * min(
            int(progbar_length*((cur_step*batch_size)/num_instances)), 
            progbar_length,
        )

    sys.stdout.write(TEMPLATE.format(
        color_code,
        cur_epoch, 
        total_epochs,
        current_progress,
        "-",
        progbar_length,
        loss, 
        acc,
        elapsed_time,
        eta,
        total,
    ))
    sys.stdout.flush()

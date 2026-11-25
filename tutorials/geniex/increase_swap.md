# Increase Swap space

Exporting an LLM requires a large amount of working memory. For large LLMs,
100+ GB can be required. Many host devices do not have 100+ GB of physical RAM.
Luckily, we can supplement this using swap space. However, you may need to
configure your system to have this much swap space.

As an example, these instructions will assume that the required amount is
150 and your physical RAM is about 32 GB, which means we need to add ~120 GB of
swap space. The exact number that we recommend will be printed when you run a
command with insufficient memory.

## Ubuntu (including WSL)

Please follow these instructions (and change `/local/mnt/swapfile` if you want
to place the swap file somewhere else):

```
sudo swapoff -a

# bs=<amount of data that can be read/write>
# count=number of <bs> to allocate for swapfile
# Total size = <bs> * <count>
#            = 1 MB * 120k = ~120GB
# Note: Update this line depending on how much swap you need.
sudo dd if=/dev/zero of=/local/mnt/swapfile bs=1M count=120k

# Set the correct permissions
sudo chmod 0600 /local/mnt/swapfile

sudo mkswap /local/mnt/swapfile  # Set up a Linux swap area
sudo swapon /local/mnt/swapfile  # Turn the swap on
```

> [!IMPORTANT]
> The above commands will not persist through a reboot.

Reference: https://askubuntu.com/questions/178712/how-to-increase-swap-space

## Windows

If you are exporting via Ubuntu on WSL (Windows Subsystem for Linux), please
follow the Ubuntu instructions instead.

On Windows, the swap space is referred to as the "paging file size". These are
the steps to change it:

 1. Open up "System" from the task bar.
 2. In the left pane, select "System".
 3. Select "About".
 4. Select "Advanced system settings".
 5. In the "Advanced" tab, select "Performance" -> "Settings...".
 6. Select the "Advanced" tab and select "Virtual memory" -> "Change...".
 7. Untick "Automatically manage paging file size for all drives"
 8. Tick "Custom size" and select a new Initial and Maximum. It is important
    that we set the maximum to at least our recommended swap space, while the
    initial you can set as you like or leave as is. As a concrete example,
    let's select initial 25000 (~25 GB) and maximum 120000 (~120 GB).
 9. **Make sure to click the "Set" button** or it will not take effect.
10. Restart Windows.


start=$SECONDS
counter=0

export MAKEFLAGS="-j `echo \`nproc\`*2/2|bc`"

make CONFIG=SmallBoomConfig

for file in /home/dwright7/Boom-proj/riscv-tools/riscv64-unknown-elf/share/riscv-tests/benchmarks/*.riscv
do
	counter=$((counter+1))
	filename=$(echo "$file" | cut -d "." -f1 | cut -d "/" -f10)
	duration=$(( SECONDS - start ))

	echo
	echo "Starting Benchmark @ $duration seconds: "
	echo "$file ... "
	
	### ./emulator-freechips.rocketchip.system-DefaultConfig /home/dwright7/Boom-proj/riscv-tools/riscv64-unknown-elf/share/riscv-tests/benchmarks/dhrystone.riscv jtag_rbb_enable=1

	./simulator-boom.system-SmallBoomConfig $file +jtag_rbb_enable=1 > changed_outputs/multi_var_sweep/128/${filename}.out 
	echo "Done!  $counter/12"
	echo	
	echo
	
done

duration=$(( SECONDS - start ))
echo "Finished. Completed in $duration seconds!"
echo
echo



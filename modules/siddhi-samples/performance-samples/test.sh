#!/bin/bash

#ant SimpleFilterSingleQueryWithDisruptorAllCfgPerformance &> logs/pref.log
#ant SimpleFilterSingleQueryWithGpuAllCfgPerformance &> logs/SimpleFilterSingleQueryWithGpuAllCfgPerformance.log

SIDDHI_ROOT=/home/prabodha/devel/research/cep/siddhi/siddhi-git-dev

JVM=/usr/bin/java


#AGENTLIB=-agentlib:hprof=cpu=samples

CLASSPATH=${SIDDHI_ROOT}/modules/siddhi-samples/performance-samples/target/build/classes:${SIDDHI_ROOT}/modules/siddhi-samples/target/lib/antlr4-runtime-4.3.jar:${SIDDHI_ROOT}/modules/siddhi-samples/target/lib/commons-cli-1.2.jar:${SIDDHI_ROOT}/modules/siddhi-samples/target/lib/disruptor-3.2.1.jar:${SIDDHI_ROOT}/modules/siddhi-samples/target/lib/log4j-1.2.14.jar:${SIDDHI_ROOT}/modules/siddhi-samples/target/lib/siddhi-core-3.0.0-SNAPSHOT.jar:${SIDDHI_ROOT}/modules/siddhi-samples/target/lib/siddhi-query-api-3.0.0-SNAPSHOT.jar:${SIDDHI_ROOT}/modules/siddhi-samples/target/lib/siddhi-query-compiler-3.0.0-SNAPSHOT.jar:${SIDDHI_ROOT}/modules/siddhi-query-api/target/siddhi-query-api-3.0.0-SNAPSHOT.jar:${SIDDHI_ROOT}/modules/siddhi-query-compiler/target/siddhi-query-compiler-3.0.0-SNAPSHOT.jar:${SIDDHI_ROOT}/modules/siddhi-gpu/target/antlr4-annotations.jar:${SIDDHI_ROOT}/modules/siddhi-gpu/target/antlr4-runtime.jar:${SIDDHI_ROOT}/modules/siddhi-gpu/target/hamcrest-core.jar:${SIDDHI_ROOT}/modules/siddhi-gpu/target/javacpp-cudaclib-linux-x86_64.jar:${SIDDHI_ROOT}/modules/siddhi-gpu/target/javacpp-cudaclib.jar:${SIDDHI_ROOT}/modules/siddhi-gpu/target/javacpp.jar:${SIDDHI_ROOT}/modules/siddhi-gpu/target/junit.jar:${SIDDHI_ROOT}/modules/siddhi-gpu/target/log4j.jar:${SIDDHI_ROOT}/modules/siddhi-gpu/target/mvel2.jar:${SIDDHI_ROOT}/modules/siddhi-gpu/target/org.abego.treelayout.core.jar:${SIDDHI_ROOT}/modules/siddhi-gpu/target/siddhi-query-api.jar:${SIDDHI_ROOT}/modules/siddhi-gpu/target/siddhi-query-compiler.jar:${SIDDHI_ROOT}/modules/siddhi-core/target/siddhi-core-3.0.0-SNAPSHOT.jar:${SIDDHI_ROOT}/modules/siddhi-samples/performance-samples/src/main/resources/log4j.properties

APP=org.wso2.siddhi.performance.ComplexFilterMultipleQueryPerformance
LOG=logs/ComplexFilterMultipleQueryPerformance

a="false"
g="true"
r=2048
t=16
b=128
c=$((5000000 * r))
echo "${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} --enable-async true --enable-gpu false  --event-count ${c} --ringbuffer-size ${r} --threadpool-size ${t} --events-per-tblock ${b} &>${LOG}_gpu_${r}_${t}_${b}.log"
#${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} --enable-async false --enable-gpu false  --event-count ${c} --ringbuffer-size ${r} --threadpool-size ${t} --events-per-tblock ${b} &>${LOG}_cpu_singlethread_${r}_${t}_${b}.log
exit
#${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} &>${LOG}
#nohup ${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} &>${LOG} &

#for r in 512 1024 2048 4096 8192 16384; do
#	for t in 2 4 8 16; do
#		for b in 64 128 256 512 1024; do
#			c=$((5000000 * r)) 
#			echo "Executing --event-count ${c} --ringbuffer-size ${r} --threadpool-size ${t} --events-per-tblock ${b}"
#			${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} --enable-gpu false --event-count ${c} --ringbuffer-size ${r} --threadpool-size ${t} --events-per-tblock ${b} &>${LOG}_cpu_${r}_${t}_${b}.log			
#			${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} --enable-gpu true  --event-count ${c} --ringbuffer-size ${r} --threadpool-size ${t} --events-per-tblock ${b} &>${LOG}_gpu_${r}_${t}_${b}.log			
#		done
#	done
#done

for r in 2048 4096 8192 16384; do
	for t in 2 4 8 16; do
		for b in 64 128 256 512 1024; do
			c=$((5000000 * r)) 
			echo "Executing --event-count ${c} --ringbuffer-size ${r} --threadpool-size ${t} --events-per-tblock ${b}"
			${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} --enable-gpu false --event-count ${c} --ringbuffer-size ${r} --threadpool-size ${t} --events-per-tblock ${b} &>${LOG}_cpu_${r}_${t}_${b}.log			
			${JVM} ${AGENTLIB} -classpath ${CLASSPATH} ${APP} --enable-gpu true  --event-count ${c} --ringbuffer-size ${r} --threadpool-size ${t} --events-per-tblock ${b} &>${LOG}_gpu_${r}_${t}_${b}.log			
		done
	done
done

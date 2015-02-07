package org.wso2.siddhi.core.gpu.query.input.stream;

import java.util.ArrayList;
import java.util.List;

import org.wso2.siddhi.core.event.MetaComplexEvent;
import org.wso2.siddhi.core.gpu.query.input.GpuProcessStreamReceiver;
import org.wso2.siddhi.core.gpu.query.processor.GpuQueryProcessor;
import org.wso2.siddhi.core.query.input.ProcessStreamReceiver;
import org.wso2.siddhi.core.query.input.stream.StreamRuntime;
import org.wso2.siddhi.core.query.input.stream.single.SingleStreamRuntime;
import org.wso2.siddhi.core.query.input.stream.single.SingleThreadEntryValveProcessor;
import org.wso2.siddhi.core.query.output.rateLimit.OutputRateLimiter;
import org.wso2.siddhi.core.query.processor.Processor;
import org.wso2.siddhi.core.query.processor.window.TimeWindowProcessor;
import org.wso2.siddhi.core.query.selector.QuerySelector;

public class GpuStreamRuntime extends SingleStreamRuntime {
    
    private GpuQueryProcessor gpuQueryProcessor;

    public GpuStreamRuntime(ProcessStreamReceiver processStreamReceiver, Processor processorChain, MetaComplexEvent metaComplexEvent) {
        super(processStreamReceiver, processorChain, metaComplexEvent);
        gpuQueryProcessor = ((GpuProcessStreamReceiver)processStreamReceiver).getGpuQueryProcessor();
    }
    
    @Override
    public List<SingleStreamRuntime> getSingleStreamRuntimes() {
        List<SingleStreamRuntime> list = new ArrayList<SingleStreamRuntime>(1);
        list.add(this);
        return list;
    }

    @Override
    public StreamRuntime clone(String key) {
        ProcessStreamReceiver clonedProcessStreamReceiver = this.processStreamReceiver.clone(key);
        SingleThreadEntryValveProcessor singleThreadEntryValveProcessor = null;
        TimeWindowProcessor windowProcessor;
        Processor clonedProcessorChain = null;
        if (processorChain != null) {
            if (!(processorChain instanceof QuerySelector || processorChain instanceof OutputRateLimiter)) {
                clonedProcessorChain = processorChain.cloneProcessor();
                if(clonedProcessorChain instanceof SingleThreadEntryValveProcessor){
                    singleThreadEntryValveProcessor = (SingleThreadEntryValveProcessor) clonedProcessorChain;
                }
            }
            Processor processor = processorChain.getNextProcessor();
            while (processor != null) {
                if (!(processor instanceof QuerySelector || processor instanceof OutputRateLimiter)) {
                    Processor clonedProcessor = processor.cloneProcessor();
                    clonedProcessorChain.setToLast(clonedProcessor);
                    if(clonedProcessor instanceof SingleThreadEntryValveProcessor){
                        singleThreadEntryValveProcessor = (SingleThreadEntryValveProcessor) clonedProcessor;
                    } else if(clonedProcessor instanceof TimeWindowProcessor){
                        windowProcessor = (TimeWindowProcessor) clonedProcessor;
                        windowProcessor.cloneScheduler((TimeWindowProcessor) processor,singleThreadEntryValveProcessor);
                    }
                }
                processor = processor.getNextProcessor();
            }
        }
        return new SingleStreamRuntime(clonedProcessStreamReceiver, clonedProcessorChain, metaComplexEvent);
    }

    @Override
    public void setCommonProcessor(Processor commonProcessor) {
        if (processorChain == null) {
            processStreamReceiver.setNext(commonProcessor);
        } else {
            processStreamReceiver.setNext(processorChain);
            processorChain.setToLast(commonProcessor);
        }
        
        ((GpuProcessStreamReceiver)processStreamReceiver).setSelectProcessor(commonProcessor);
    }
}

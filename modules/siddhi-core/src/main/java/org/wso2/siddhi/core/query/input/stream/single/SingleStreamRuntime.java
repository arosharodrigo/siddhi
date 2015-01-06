/*
 * Copyright (c) 2005 - 2014, WSO2 Inc. (http://www.wso2.org)
 * All Rights Reserved.
 *
 * WSO2 Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.wso2.siddhi.core.query.input.stream.single;

import org.wso2.siddhi.core.event.MetaComplexEvent;
import org.wso2.siddhi.core.query.input.ProcessStreamReceiver;
import org.wso2.siddhi.core.query.input.stream.StreamRuntime;
import org.wso2.siddhi.core.query.output.rateLimit.OutputRateLimiter;
import org.wso2.siddhi.core.query.processor.Processor;
import org.wso2.siddhi.core.query.processor.window.TimeWindowProcessor;
import org.wso2.siddhi.core.query.selector.QuerySelector;

import java.util.ArrayList;
import java.util.List;

public class SingleStreamRuntime implements StreamRuntime {

    private Processor processorChain;
    private MetaComplexEvent metaComplexEvent;
    private ProcessStreamReceiver processStreamReceiver;

    public SingleStreamRuntime(ProcessStreamReceiver processStreamReceiver, Processor processorChain, MetaComplexEvent metaComplexEvent) {
        this.processStreamReceiver = processStreamReceiver;
        this.processorChain = processorChain;
        this.metaComplexEvent = metaComplexEvent;
    }

    public Processor getProcessorChain() {
        return processorChain;
    }

    public void setProcessorChain(Processor processorChain) {
        this.processorChain = processorChain;
    }

    public ProcessStreamReceiver getProcessStreamReceiver() {
        return processStreamReceiver;
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
    }

    public MetaComplexEvent getMetaComplexEvent() {
        return metaComplexEvent;
    }

    @Override
    public void configureRuntime(MetaComplexEvent metaComplexEvent) {
        if(processorChain != null) {
            processorChain.configureProcessor(metaComplexEvent);
        }
    }
}

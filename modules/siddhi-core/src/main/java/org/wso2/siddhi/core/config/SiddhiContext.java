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

package org.wso2.siddhi.core.config;

import org.wso2.siddhi.core.util.SiddhiExtensionLoader;

import java.util.Map;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadPoolExecutor;

public class SiddhiContext {

    private int eventBufferSize;
    private int threadPoolInitSize;
    private Map<String, Class> siddhiExtensions;
    private ThreadPoolExecutor executorService = null;
    private ScheduledExecutorService scheduledExecutorService = null;

    public SiddhiContext() {
        setSiddhiExtensions(SiddhiExtensionLoader.loadSiddhiExtensions());
        threadPoolInitSize = 0;
    }

    public int getEventBufferSize() {
        return eventBufferSize;
    }

    public Map<String, Class> getSiddhiExtensions() {
        return siddhiExtensions;
    }

    public void setSiddhiExtensions(Map<String, Class> siddhiExtensions) {
        this.siddhiExtensions = siddhiExtensions;
    }

    public void setEventBufferSize(int eventBufferSize) {
        this.eventBufferSize = eventBufferSize;
    }
    
    public ScheduledExecutorService getScheduledExecutorService() {
        return scheduledExecutorService;
    }

    public void setScheduledExecutorService(ScheduledExecutorService scheduledExecutorService) {
        this.scheduledExecutorService = scheduledExecutorService;
    }
    
    public void setExecutorService(ThreadPoolExecutor executorService) {
        this.executorService = executorService;
    }

    public ThreadPoolExecutor getExecutorService() {
        return executorService;
    }

    public int getThreadPoolInitSize() {
        return threadPoolInitSize;
    }

    public void setThreadPoolInitSize(int threadPoolInitSize) {
        this.threadPoolInitSize = threadPoolInitSize;
    }
}

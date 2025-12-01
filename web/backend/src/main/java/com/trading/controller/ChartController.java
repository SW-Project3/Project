package com.trading.controller;

import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;
import org.springframework.http.MediaType;

import java.util.HashMap;
import java.util.Map;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;

@RestController
@RequestMapping("/api/chart")
@CrossOrigin(origins = "*")
public class ChartController {

    @GetMapping("/data")
    public ResponseEntity<Map<String, Object>> getChartData(
            @RequestParam(required = false) String startDate,
            @RequestParam(required = false) String endDate) {
        
        Map<String, Object> response = new HashMap<>();
        
        // 기본 날짜 범위: 9/1 - 10/31
        if (startDate == null || startDate.isEmpty()) {
            startDate = "2024-09-01";
        }
        if (endDate == null || endDate.isEmpty()) {
            endDate = "2024-10-31";
        }
        
        response.put("startDate", startDate);
        response.put("endDate", endDate);
        response.put("message", "Chart data endpoint - Python backend will handle data processing");
        
        return ResponseEntity.ok(response);
    }
}


package com.tfg.brain_to_text_web.controller;

import com.tfg.brain_to_text_web.model.Prediction;
import com.tfg.brain_to_text_web.model.Trial;
import com.tfg.brain_to_text_web.service.TrialService;
import jakarta.servlet.http.HttpSession;
import java.util.Optional;
import org.springframework.data.domain.Page;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class HomeController {

    private final TrialService trialService;

    public HomeController(TrialService trialService) {
        this.trialService = trialService;
    }

    @GetMapping("/")
    public String index() {
        return "index";
    }

    @GetMapping("/trials")
    public String trials(
            @RequestParam(defaultValue = "all") String split,
            @RequestParam(defaultValue = "0") int page,
            Model model
    ) {
        Page<Trial> trialPage = trialService.findTrialsPage(split, page, 8);
        model.addAttribute("trials", trialPage.getContent());
        model.addAttribute("trialPage", trialPage);
        model.addAttribute("splits", trialService.findSplits());
        model.addAttribute("selectedSplit", split);
        return "trials";
    }

    @GetMapping("/trials/{id}")
    public String detail(
            @PathVariable Long id,
            HttpSession session,
            Model model
    ) {
        Trial trial = trialService.findTrialWithPredictions(id);
        String modelName = (String) session.getAttribute(modelSessionKey(id));
        if (modelName == null || modelName.isBlank()) {
            modelName = "baseline";
        }
        String selectedModelName = modelName;

        Optional<Prediction> selectedPrediction = trial.getPredictions().stream()
                .filter(prediction -> prediction.getModelName().equals(selectedModelName))
                .findFirst();

        model.addAttribute("trial", trial);
        model.addAttribute("selectedModel", selectedModelName);
        model.addAttribute("prediction", selectedPrediction.orElse(null));
        return "trial-detail";
    }

    @PostMapping("/trials/{id}/model")
    public String selectModel(
            @PathVariable Long id,
            @RequestParam String modelName,
            HttpSession session
    ) {
        if (!modelName.equals("baseline") && !modelName.equals("proposed")) {
            modelName = "baseline";
        }
        session.setAttribute(modelSessionKey(id), modelName);
        return "redirect:/trials/" + id;
    }

    private String modelSessionKey(Long trialId) {
        return "selectedModel:" + trialId;
    }
}
